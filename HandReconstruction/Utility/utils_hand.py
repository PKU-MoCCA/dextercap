import numpy as np
import rerun as rr
import smplx
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

import HandReconstruction.config as config
from HandReconstruction.Utility.utils_pytorch3d import (
    euler_angles_to_matrix,
    matrix_to_axis_angle,
)
from HandReconstruction.Utility.utils_visualize import init_rerun, visualize_hand

# Index finger, middle finger, pinky finger, ring finger, thumb
#  transform order of right hand
#         15-14-13-\
#                   \
#    3-- 2 -- 1 -----0
#   6 -- 5 -- 4 ----/
#   12 - 11 - 10 --/
#    9-- 8 -- 7 --/

device = torch.device(config.optimize["device"])

# DOF related variables
dof = torch.tensor(config.hand["dof"], device=device)  # (15,) not include root
num_dof = torch.sum(dof)
index_dof = torch.cumsum(dof, dim=0) - dof  # (15,) not include root
manual_angle = [18, -10, 0, 0, 5]  # Thumb, index, middle, ring, pinky

limit_thumb_MCP_bend = torch.tensor([-np.pi / 2, np.pi / 6], device=device)
limit_thumb_MCP_spread = torch.tensor([-np.pi / 6, np.pi / 6], device=device)
limit_thumb_MCP_twist = torch.tensor([-np.pi / 4, np.pi / 4], device=device)
limit_thumb_PIP_bend = torch.tensor([-np.pi / 3, 0], device=device)
limit_thumb_PIP_spread = torch.tensor([-np.pi / 6, np.pi / 6], device=device)
limit_thumb_PIP_twist = torch.tensor([-np.pi / 6, np.pi / 6], device=device)
limit_thumb_DIP_bend = torch.tensor([-np.pi / 2, 0], device=device)

limit_index_MCP_bend = torch.tensor([-np.pi / 2, np.pi / 6], device=device)
limit_index_MCP_spread = torch.tensor([-np.pi / 6, np.pi / 6], device=device)
limit_index_MCP_twist = torch.tensor([-np.pi / 6, np.pi / 6], device=device)
limit_index_PIP_bend = torch.tensor([-np.pi / 2, 0], device=device)
limit_index_DIP_bend = torch.tensor([-np.pi / 2, 0], device=device)

limit_middle_MCP_bend = torch.tensor([-np.pi / 2, np.pi / 6], device=device)
limit_middle_MCP_spread = torch.tensor([-np.pi / 6, np.pi / 6], device=device)
limit_middle_MCP_twist = torch.tensor([-np.pi / 6, np.pi / 6], device=device)
limit_middle_PIP_bend = torch.tensor([-np.pi / 2, 0], device=device)
limit_middle_DIP_bend = torch.tensor([-np.pi / 2, 0], device=device)

limit_ring_MCP_bend = torch.tensor([-np.pi / 2, np.pi / 6], device=device)
limit_ring_MCP_spread = torch.tensor([-np.pi / 6, np.pi / 6], device=device)
limit_ring_MCP_twist = torch.tensor([-np.pi / 6, np.pi / 6], device=device)
limit_ring_PIP_bend = torch.tensor([-np.pi / 2, 0], device=device)
limit_ring_DIP_bend = torch.tensor([-np.pi / 2, 0], device=device)

limit_pinky_MCP_bend = torch.tensor([-np.pi / 2, np.pi / 6], device=device)
limit_pinky_MCP_spread = torch.tensor([-np.pi / 6, np.pi / 6], device=device)
limit_pinky_MCP_twist = torch.tensor([-np.pi / 6, np.pi / 6], device=device)
limit_pinky_PIP_bend = torch.tensor([-np.pi / 2, 0], device=device)
limit_pinky_DIP_bend = torch.tensor([-np.pi / 2, 0], device=device)


# 指关节顺序：食指、中指、小指、无名指、拇指
joints_name_list = ["index", "middle", "pinky", "ring", "thumb"]

# 关节限制列表
dof_limit_list = []
for i, joint in enumerate(joints_name_list):
    # MCP
    if dof[3 * i] >= 3:
        dof_limit_list.append(eval(f"limit_{joint}_MCP_twist"))  # X
    if dof[3 * i] >= 2:
        dof_limit_list.append(eval(f"limit_{joint}_MCP_spread"))  # Y
    if dof[3 * i] >= 1:
        dof_limit_list.append(eval(f"limit_{joint}_MCP_bend"))  # Z
    # PIP
    if dof[3 * i + 1] >= 3:
        dof_limit_list.append(eval(f"limit_{joint}_PIP_twist"))
    if dof[3 * i + 1] >= 2:
        dof_limit_list.append(eval(f"limit_{joint}_PIP_spread"))
    if dof[3 * i + 1] >= 1:
        dof_limit_list.append(eval(f"limit_{joint}_PIP_bend"))
    # DIP
    if dof[3 * i + 2] >= 3:
        dof_limit_list.append(eval(f"limit_{joint}_DIP_twist"))
    if dof[3 * i + 2] >= 2:
        dof_limit_list.append(eval(f"limit_{joint}_DIP_spread"))
    if dof[3 * i + 2] >= 1:
        dof_limit_list.append(eval(f"limit_{joint}_DIP_bend"))

dof_limit = torch.stack(dof_limit_list, dim=0)


def get_relative_rotations():
    """
    Calculate and cache relative rotations between joints in their neutral pose.

    Returns:
        torch.Tensor: Tensor of relative rotation matrices, shape=(16, 3, 3)
    """
    # Define joint indices
    ROOT_index = 0
    MCP_index = torch.tensor([13, 1, 4, 10, 7])
    PIP_index = torch.tensor([14, 2, 5, 11, 8])
    DIP_index = torch.tensor([15, 3, 6, 12, 9])

    # Get MANO model and joints for neutral pose
    smplx_model = smplx.create(
        model_path=config.data["smpl_model_path"],
        model_type="mano",
        flat_hand_mean=True,
        is_rhand=False,
        use_pca=False,
        batch_size=1,
    )

    # Get joints in neutral pose
    hand_parms = {
        "global_orient": torch.zeros((1, 3)),
        "transl": torch.zeros((1, 3)),
        "hand_pose": torch.zeros((1, 45)),
        "betas": torch.zeros((1, 10)),
    }
    smplx_output = smplx_model(**hand_parms)
    joints = smplx_output.joints[0].detach()

    # Calculate bone directions
    MCP_bone = joints[MCP_index] - joints[ROOT_index]
    MCP_bone = F.normalize(MCP_bone, dim=1)
    PIP_bone = joints[PIP_index] - joints[MCP_index]
    PIP_bone = F.normalize(PIP_bone, dim=1)
    DIP_bone = joints[DIP_index] - joints[PIP_index]
    DIP_bone = F.normalize(DIP_bone, dim=1)

    # Calculate palm normals
    palm_n = torch.linalg.cross(MCP_bone[:-1], MCP_bone[1:], dim=1)
    palm_n = F.normalize(palm_n, dim=1)

    # Initialize coordinate systems
    device = torch.device(config.optimize["device"])
    local_coords = torch.zeros((16, 3, 3), dtype=torch.float32, device=device)
    local_coords[0] = torch.eye(3, device=device)
    relative_rotations = torch.zeros((16, 3, 3), dtype=torch.float32, device=device)
    relative_rotations[0] = torch.eye(3, device=device)

    # Set up coordinate systems and calculate relative rotations
    for i, (mcp, pip, dip) in enumerate(zip(MCP_index, PIP_index, DIP_index)):
        # MCP joints
        local_coords[mcp, :, 0] = PIP_bone[i]
        if i == 0:
            local_coords[mcp, :, 1] = palm_n[0]
        elif i == 4:
            local_coords[mcp, :, 1] = palm_n[3]
        else:
            palm_n_i = (palm_n[i - 1] + palm_n[i]) / 2
            local_coords[mcp, :, 1] = palm_n_i

        # 使用手动设定的角度校准, 将local_coords[mcp, :, 1]绕着local_coords[mcp, :, 0]为轴逆时针旋转, 手动设置角度
        angle_rad = np.radians(manual_angle[i])
        axis = local_coords[mcp, :, 0].cpu().numpy()
        vector = local_coords[mcp, :, 1].cpu().numpy()
        rot = R.from_rotvec(angle_rad * axis)
        rotated_vector = rot.apply(vector)
        local_coords[mcp, :, 1] = torch.tensor(rotated_vector, device=device)

        # Gram-Schmidt for MCP
        local_coords[mcp, :, 0] = F.normalize(local_coords[mcp, :, 0], dim=0)
        local_coords[mcp, :, 1] = (
            local_coords[mcp, :, 1]
            - torch.sum(local_coords[mcp, :, 1] * local_coords[mcp, :, 0])
            * local_coords[mcp, :, 0]
        )
        local_coords[mcp, :, 1] = F.normalize(local_coords[mcp, :, 1], dim=0)
        local_coords[mcp, :, 2] = torch.linalg.cross(
            local_coords[mcp, :, 0], local_coords[mcp, :, 1], dim=0
        )
        relative_rotations[mcp] = local_coords[0].T @ local_coords[mcp]

        # PIP joints
        local_coords[pip, :, 0] = DIP_bone[i]
        if i == 0:
            local_coords[pip, :, 1] = palm_n[0]
        elif i == 4:
            local_coords[pip, :, 1] = palm_n[3]
        else:
            palm_n_i = (palm_n[i - 1] + palm_n[i]) / 2
            local_coords[pip, :, 1] = palm_n_i

        # 使用手动设定的角度校准, 将local_coords[pip, :, 1]绕着local_coords[pip, :, 0]为轴逆时针旋转, 手动设置角度
        angle_rad = np.radians(manual_angle[i])
        axis = local_coords[pip, :, 0].cpu().numpy()
        vector = local_coords[pip, :, 1].cpu().numpy()
        rot = R.from_rotvec(angle_rad * axis)
        rotated_vector = rot.apply(vector)
        local_coords[pip, :, 1] = torch.tensor(rotated_vector, device=device)

        # Gram-Schmidt for PIP
        local_coords[pip, :, 0] = F.normalize(local_coords[pip, :, 0], dim=0)
        local_coords[pip, :, 1] = (
            local_coords[pip, :, 1]
            - torch.sum(local_coords[pip, :, 1] * local_coords[pip, :, 0])
            * local_coords[pip, :, 0]
        )
        local_coords[pip, :, 1] = F.normalize(local_coords[pip, :, 1], dim=0)
        local_coords[pip, :, 2] = torch.linalg.cross(
            local_coords[pip, :, 0], local_coords[pip, :, 1], dim=0
        )
        relative_rotations[pip] = local_coords[mcp].T @ local_coords[pip]

        # DIP joints
        local_coords[dip, :, 0] = DIP_bone[i]
        if i == 0:
            local_coords[dip, :, 1] = palm_n[0]
        elif i == 4:
            local_coords[dip, :, 1] = palm_n[3]
        else:
            palm_n_i = (palm_n[i - 1] + palm_n[i]) / 2
            local_coords[dip, :, 1] = palm_n_i

        # 使用手动设定的角度校准, 将local_coords[dip, :, 1]绕着local_coords[dip, :, 0]为轴逆时针旋转, 手动设置角度
        angle_rad = np.radians(manual_angle[i])
        axis = local_coords[dip, :, 0].cpu().numpy()
        vector = local_coords[dip, :, 1].cpu().numpy()
        rot = R.from_rotvec(angle_rad * axis)
        rotated_vector = rot.apply(vector)
        local_coords[dip, :, 1] = torch.tensor(rotated_vector, device=device)

        # Gram-Schmidt for DIP
        local_coords[dip, :, 0] = F.normalize(local_coords[dip, :, 0], dim=0)
        local_coords[dip, :, 1] = (
            local_coords[dip, :, 1]
            - torch.sum(local_coords[dip, :, 1] * local_coords[dip, :, 0])
            * local_coords[dip, :, 0]
        )
        local_coords[dip, :, 1] = F.normalize(local_coords[dip, :, 1], dim=0)
        local_coords[dip, :, 2] = torch.linalg.cross(
            local_coords[dip, :, 0], local_coords[dip, :, 1], dim=0
        )
        relative_rotations[dip] = local_coords[pip].T @ local_coords[dip]

    return relative_rotations


# @torch.jit.script
def dof_to_rot_vector(
    dof_value: torch.Tensor,
    dof: torch.Tensor = torch.tensor(config.hand["dof"], device=device),
) -> torch.Tensor:
    """
    Convert biomechanical degrees of freedom (DOF) values to rotation vectors for the MANO hand model.

    Parameters:
        dof_value (torch.Tensor): Tensor containing joint angles for each DOF, shape=(batch_size, num_dofs_pose)
        dof (torch.Tensor): Tensor containing the DOF of the MANO hand model, shape=(15,)

    Returns:
        torch.Tensor: Rotation vectors for each joint in the MANO model format, shape=(batch_size, 16, 3)
    """
    batch_size = dof_value.shape[0]
    device = dof_value.device

    # Pre-printed cached relative rotations from function get_relative_rotations
    relative_rotations = [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [
            [0.990314781665802, -0.13280899822711945, -0.04047652706503868],
            [0.12150106579065323, 0.9700703620910645, -0.21023999154567719],
            [0.06718684732913971, 0.2032858282327652, 0.9768114686012268],
        ],
        [
            [0.9811518788337708, 0.1903766244649887, 0.03312995657324791],
            [-0.19037672877311707, 0.981705904006958, -0.0031802207231521606],
            [-0.033129315823316574, -0.0031869113445281982, 0.999445915222168],
        ],
        [
            [1.0, 1.210719347000122e-08, 4.656612873077393e-10],
            [1.210719347000122e-08, 1.0, -1.4901161193847656e-08],
            [4.656612873077393e-10, -1.4901161193847656e-08, 0.9999999403953552],
        ],
        [
            [0.982589840888977, -0.05310700833797455, 0.17803621292114258],
            [0.058651916682720184, 0.9979392290115356, -0.02602403052151203],
            [-0.17628726363182068, 0.03601311147212982, 0.9836797714233398],
        ],
        [
            [0.9936535358428955, 0.1120494157075882, -0.009878471493721008],
            [-0.11204799264669418, 0.9937025904655457, 0.000699702650308609],
            [0.00989466905593872, 0.00041159987449645996, 0.9999510049819946],
        ],
        [
            [1.0, 4.190951585769653e-09, 0.0],
            [4.190951585769653e-09, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        [
            [0.8059144020080566, -0.11678124964237213, 0.5803999900817871],
            [0.0029284136835485697, 0.9811265468597412, 0.19334447383880615],
            [-0.5920248627662659, -0.15411943197250366, 0.7910460233688354],
        ],
        [
            [0.9976354241371155, 0.04977748915553093, -0.04739096760749817],
            [-0.04954490810632706, 0.9987534284591675, 0.006070777773857117],
            [0.0476340651512146, -0.00370846688747406, 0.9988579750061035],
        ],
        [[1.0, 0.0, 0.0], [0.0, 0.9999999403953552, 0.0], [0.0, 0.0, 1.0]],
        [
            [0.9816410541534424, -0.10613077133893967, 0.15848369896411896],
            [0.0716620683670044, 0.9752410054206848, 0.20921172201633453],
            [-0.1767635941505432, -0.1940135508775711, 0.9649421572685242],
        ],
        [
            [0.9810875058174133, 0.10822392255067825, 0.16048318147659302],
            [-0.10737982392311096, 0.9941202402114868, -0.013949096202850342],
            [-0.16104920208454132, -0.003547370433807373, 0.9869400262832642],
        ],
        [
            [1.0, 0.0, -2.9802322387695312e-08],
            [0.0, 1.0, 0.0],
            [-2.9802322387695312e-08, 0.0, 1.0],
        ],
        [
            [0.6376827359199524, -0.5680447816848755, -0.5202747583389282],
            [0.028948858380317688, 0.6926146149635315, -0.7207266688346863],
            [0.769754946231842, 0.4445336163043976, 0.4581126272678375],
        ],
        [
            [0.9297419786453247, 0.36749404668807983, 0.022979170083999634],
            [-0.366394579410553, 0.9295425415039062, -0.0412992388010025],
            [-0.036537349224090576, 0.029978185892105103, 0.998882532119751],
        ],
        [
            [1.0000001192092896, 2.9802322387695312e-08, 0.0],
            [2.9802322387695312e-08, 0.9999998807907104, 0.0],
            [0.0, 0.0, 1.0],
        ],
    ]
    relative_rotations = torch.tensor(relative_rotations, device=device)

    # Pre-defined parent index from MANO model
    parent_index = torch.tensor(
        [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14], device=device
    )

    # Initialize tensors
    local_coords = torch.zeros((batch_size, 16, 3, 3), device=device)
    local_coords[:, 0] = torch.eye(3, device=device)
    local_coords_after_rotation = torch.zeros((batch_size, 16, 3, 3), device=device)
    local_coords_after_rotation[:, 0] = torch.eye(3, device=device)
    joint_orientations = torch.zeros((batch_size, 16, 3, 3), device=device)
    joint_orientations[:, 0] = torch.eye(3, device=device)
    all_rot_mat = torch.zeros((batch_size, 16, 3, 3), device=device)
    all_rot_mat[:, 0] = torch.eye(3, device=device)

    value_idx = 0
    for i in range(16):
        if i == 0:
            continue
        parent = parent_index[i]

        if i == 0 or dof[i - 1] == 3:
            angles = dof_value[:, value_idx : value_idx + 3]
            rot_mat = euler_angles_to_matrix(angles, "XYZ")
            value_idx += 3
        elif dof[i - 1] == 1:
            z_angle = dof_value[:, value_idx : value_idx + 1]
            rot_mat = euler_angles_to_matrix(
                torch.cat(
                    [torch.zeros_like(z_angle), torch.zeros_like(z_angle), z_angle],
                    dim=1,
                ),
                "XYZ",
            )
            value_idx += 1
        elif dof[i - 1] == 2:
            y_angle = dof_value[:, value_idx : value_idx + 1]
            z_angle = dof_value[:, value_idx + 1 : value_idx + 2]
            rot_mat = euler_angles_to_matrix(
                torch.cat([torch.zeros_like(y_angle), y_angle, z_angle], dim=1), "XYZ"
            )
            value_idx += 2
        else:
            # Default case: identity matrix if none of the conditions match
            raise ValueError(f"Unexpected DOF value at joint {i - 1}: {dof[i - 1]}")

        # Update rotations without in-place operations
        local_coords[:, i] = (
            local_coords_after_rotation[:, parent] @ relative_rotations[i]
        )
        local_coords_after_rotation[:, i] = local_coords[:, i].clone() @ rot_mat

        # Create new tensor for joint orientations instead of modifying in-place
        parent_orientation = joint_orientations[:, parent].clone()
        joint_orientations[:, i] = parent_orientation @ rot_mat
        all_rot_mat[:, i] = rot_mat

    # Compute final rotations with explicit new tensor creation
    parent_orientations = joint_orientations[:, parent_index].clone()
    parent_orientations_inv = parent_orientations.transpose(-2, -1)

    local_rotations = (
        local_coords.clone() @ all_rot_mat @ local_coords.transpose(-2, -1)
    )

    rot_matrix = parent_orientations_inv @ local_rotations @ parent_orientations

    # Convert to axis-angle
    rot_vectors = matrix_to_axis_angle(rot_matrix)
    rot_vectors = rot_vectors.reshape(batch_size, 16, 3)[:, 1:]

    return rot_vectors, local_coords


def main():
    # relative_rotations = get_relative_rotations()
    # print(relative_rotations.detach().cpu().numpy().tolist())

    init_rerun("MANO_test_rotation", save=False)

    # Define test hand parameters
    hand_translation1 = np.zeros((1, 3))
    hand_orientation1 = np.zeros((1, 3))
    hand_pose1 = np.zeros((1, 45))
    hand_shape1 = np.zeros((1, 10))

    # # Visualize initial neutral pose
    # visualize_hand(
    #     hand_translation=hand_translation1,
    #     hand_orientation=hand_orientation1,
    #     hand_pose=hand_pose1,
    #     hand_shape=hand_shape1,
    #     hand_name="hand1",
    #     visualize_joint=True,
    #     visualize_joint_label=True,
    # )

    # Test pose: make a fist-like gesture
    dof_value = [
        [0, -np.pi / 6, 0],  # Index MCP, twist, spread, bend
        [0],  # Index PIP
        [0],  # Index DIP
        [0, 0, -np.pi / 6],  # Middle MCP
        [-np.pi / 3],  # Middle PIP
        [-np.pi / 3],  # Middle DIP
        [0, 0, -np.pi / 2],  # Pinky MCP
        [0],  # Pinky PIP
        [0],  # Pinky DIP
        [0, 0, 0],  # Ring MCP
        [-np.pi / 2],  # Ring PIP
        [-np.pi / 2],  # Ring DIP
        [np.pi / 6, -np.pi / 6, -np.pi / 3],  # Thumb MCP
        [0, 0, 0],  # Thumb PIP
        [0],  # Thumb DIP
    ]  # Thumb DIP
    dof_value = np.concatenate(dof_value, axis=0)
    # dof_value = np.zeros_like(dof_value)

    # Convert DOF values to rotation vectors
    rot_vectors, local_coords = dof_to_rot_vector(
        torch.tensor(dof_value, dtype=torch.float32).unsqueeze(0)
    )

    # Extract the first batch since we only have one sample
    local_coords = local_coords[0].detach().cpu().numpy()  # Now shape is (16, 3, 3)

    # Convert to MANO format
    hand_translation2 = np.array([[0, 0, 0]])  # Offset from first hand
    hand_orientation2 = np.array([[0, 1, 0]])  # Root joint rotation
    hand_pose2 = (
        rot_vectors.detach().cpu().numpy().reshape(1, -1)
    )  # Other joint rotations

    # Visualize posed hand
    visualize_hand(
        hand_translation=hand_translation2,
        hand_orientation=hand_orientation2,
        hand_pose=hand_pose2,
        hand_shape=hand_shape1,
        hand_name="hand2",
        visualize_joint=True,
        visualize_joint_label=True,
    )

    # Get MANO model and joints for visualization
    from HandReconstruction import config

    smplx_model = smplx.create(
        model_path=config.data["smpl_model_path"],
        model_type="mano",
        flat_hand_mean=True,
        is_rhand=False,
        use_pca=False,
        batch_size=1,
    )

    # Get joints for posed hand
    hand_parms2 = {
        "global_orient": torch.tensor(hand_orientation2, dtype=torch.float32),
        "transl": torch.tensor(hand_translation2, dtype=torch.float32),
        "hand_pose": torch.tensor(hand_pose2, dtype=torch.float32),
        "betas": torch.tensor(hand_shape1, dtype=torch.float32),
    }
    smplx_output2 = smplx_model(**hand_parms2)
    joints2 = smplx_output2.joints[0].detach().cpu().numpy()

    R_global = R.from_rotvec(hand_orientation2)
    for i in range(16):
        local_coords[i] = R_global.apply(local_coords[i].T)

    # Visualize coordinate systems
    for i in range(16):
        rr.log(
            f"hand2/coords/joint_{i}",
            rr.Arrows3D(
                origins=np.tile(joints2[i], (3, 1)),
                vectors=local_coords[i] * 0.02,
                radii=0.001,
                colors=np.array(
                    [[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8
                ),
            ),
        )

    # def is_orthogonal(matrix: torch.Tensor) -> bool:
    #     identity = torch.eye(3, device=matrix.device)
    #     return torch.allclose(
    #         matrix.T @ matrix, identity, atol=1e-6
    #     ) and torch.allclose(matrix @ matrix.T, identity, atol=1e-6)

    # for i in range(16):
    #     if not is_orthogonal(local_coords[i]):
    #         print(f"Joint {i} local_coords is not orthogonal.")


if __name__ == "__main__":
    main()
