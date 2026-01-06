import torch
import torch.nn.functional as F
from torch import jit


@jit.script
def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


@jit.script
def _copysign(a, b):
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


@jit.script
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions.

    Args:
        matrix (torch.Tensor): Rotation matrices with shape (..., 3, 3)

    Returns:
        torch.Tensor: Quaternions with shape (..., 4)
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape f{matrix.shape}.")

    # 扁平化批处理维度
    original_shape = matrix.shape[:-2]
    matrix_flat = matrix.reshape(-1, 3, 3)

    # 提取矩阵元素
    m00 = matrix_flat[:, 0, 0]
    m01 = matrix_flat[:, 0, 1]
    m02 = matrix_flat[:, 0, 2]
    m10 = matrix_flat[:, 1, 0]
    m11 = matrix_flat[:, 1, 1]
    m12 = matrix_flat[:, 1, 2]
    m20 = matrix_flat[:, 2, 0]
    m21 = matrix_flat[:, 2, 1]
    m22 = matrix_flat[:, 2, 2]

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    min_value = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(min_value))

    # 获取结果并恢复原始批处理维度
    flat_result = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ]

    return flat_result.reshape(original_shape + (4,))


@torch.jit.script
def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Create rotation matrix from axis and angle.

    Parameters:
        axis (str): One of 'X', 'Y', or 'Z'
        angle (torch.Tensor): Rotation angle in radians, shape=(...)

    Returns:
        torch.Tensor: Rotation matrix, shape=(..., 3, 3)
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    # Create shape (..., 3, 3) identity matrices
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError(f"Invalid axis: {axis}")

    # Stack and reshape to (..., 3, 3)
    shape = angle.shape + (3, 3)
    return torch.stack(R_flat, dim=-1).reshape(shape)


@torch.jit.script
def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert euler angles to rotation matrix.

    Parameters:
        euler_angles (torch.Tensor): Euler angles in radians, shape=(..., 3)
        convention (str): Convention string made up of 'X', 'Y', and 'Z'

    Returns:
        torch.Tensor: Rotation matrices, shape=(..., 3, 3)
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")

    # TorchScript compatible implementation without using map()
    result_mat = torch.eye(3, device=euler_angles.device)
    angles = torch.unbind(euler_angles, -1)

    for i, letter in enumerate(convention):
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")

        angle = angles[i]
        rot_mat = _axis_angle_rotation(letter, angle)
        result_mat = torch.matmul(rot_mat, result_mat)

    return result_mat


@jit.script
def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


@jit.script
def _index_from_letter(letter: str) -> int:
    """
    Convert letter (X, Y, Z) to corresponding index (0, 1, 2).

    Args:
        letter (str): One of ['X', 'Y', 'Z']

    Returns:
        int: Index corresponding to the letter (0 for X, 1 for Y, 2 for Z)
    """
    if letter == "X":
        return 0
    elif letter == "Y":
        return 1
    elif letter == "Z":
        return 2
    else:
        # 提供默认值以确保始终返回一个整数
        return -1


@jit.script
def matrix_to_euler_angles(matrix, convention: str):
    """
    Convert rotation matrices to euler angles in the specified convention.

    Args:
        matrix (torch.Tensor): Rotation matrices with shape (..., 3, 3)
        convention (str): Convention string made up of 'X', 'Y', and 'Z'

    Returns:
        torch.Tensor: Euler angles in radians with shape (..., 3)
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


@jit.script
def standardize_quaternion(quaternions):
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


@jit.script
def quaternion_raw_multiply(a, b):
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


@jit.script
def quaternion_multiply(a, b):
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


@jit.script
def quaternion_invert(quaternion):
    """
    Invert a quaternion.

    Args:
        quaternion (torch.Tensor): Quaternion with shape (..., 4)

    Returns:
        torch.Tensor: Inverted quaternion with shape (..., 4)
    """
    # 创建常量张量 [1, -1, -1, -1]，使用与输入相同的设备和数据类型
    invert_mask = torch.tensor(
        [1.0, -1.0, -1.0, -1.0], dtype=quaternion.dtype, device=quaternion.device
    )
    return quaternion * invert_mask


@jit.script
def quaternion_apply(quaternion, point):
    """
    Apply quaternion rotation to points.

    Args:
        quaternion (torch.Tensor): Quaternion with shape (..., 4)
        point (torch.Tensor): Points with shape (..., 3)

    Returns:
        torch.Tensor: Rotated points with shape (..., 3)
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    # 使用 zeros 函数而不是 new_zeros
    real_parts = torch.zeros(
        point.shape[:-1] + (1,), dtype=point.dtype, device=point.device
    )
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


@jit.script
def axis_angle_to_quaternion(axis_angle):
    """
    Convert axis-angle representation to quaternion.

    Args:
        axis_angle (torch.Tensor): Axis-angle representation with shape (..., 3)

    Returns:
        torch.Tensor: Quaternion with shape (..., 4)
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


@jit.script
def axis_angle_to_matrix(axis_angle):
    """
    Convert axis-angle representation to rotation matrix.

    Args:
        axis_angle (torch.Tensor): Axis-angle representation with shape (..., 3)

    Returns:
        torch.Tensor: Rotation matrices with shape (..., 3, 3)
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


@jit.script
def quaternion_to_axis_angle(quaternions):
    """
    Convert quaternion to axis-angle representation.

    Args:
        quaternions (torch.Tensor): Quaternion with shape (..., 4)

    Returns:
        torch.Tensor: Axis-angle representation with shape (..., 3)
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


@jit.script
def matrix_to_axis_angle(matrix):
    """
    Convert rotation matrix to axis-angle representation.

    Args:
        matrix (torch.Tensor): Rotation matrix with shape (..., 3, 3)

    Returns:
        torch.Tensor: Axis-angle representation with shape (..., 3)
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


@jit.script
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


@jit.script
def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to 6D rotation representation.

    Args:
        matrix (torch.Tensor): Rotation matrix with shape (..., 3, 3)

    Returns:
        torch.Tensor: 6D rotation representation with shape (..., 6)
    """
    batch_dims = matrix.shape[:-2]
    first_two_columns = matrix[..., :2, :]
    return first_two_columns.clone().reshape(batch_dims + (6,))
