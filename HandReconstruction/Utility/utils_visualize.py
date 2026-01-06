import os

import numpy as np
import rerun as rr
import smplx
import torch
import trimesh
from scipy.spatial.transform import Rotation as R

import HandReconstruction.config as config
from HandReconstruction.Data.mano_segment import vertice_of_patch
from HandReconstruction.Utility.utils_mesh import (
    compute_vertex_normals,
    sample_point_cloud_from_mesh,
)


def init_rerun(path: str, save: bool = True):
    """
    Initialize the rerun visualization.

    Parameters:
    - path (str): The path of the rerun session, e.g. `Result/RerunSession/<timestamp>/<frame_count>.rrd`.
    - save (bool): Whether to save the rerun session. Default is True.
        - If True, the rerun session will be saved in the `path`.
        - If False, the rerun session will not be saved, but the visualization will be directly shown in the browser.
    """

    folder_path = os.path.dirname(path)
    file_name = os.path.basename(path)
    rr.init(file_name, spawn=not save)
    if save:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        rr.save(path)
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # Set an up-axis = +Y
    rr.set_time("stable_time", duration=0)


def visualize_hand(
    hand_translation: np.ndarray,
    hand_orientation: np.ndarray,
    hand_pose: np.ndarray,
    hand_shape: np.ndarray,
    hand_name: str = "hand",
    flat_hand_mean: bool = True,
    visualize_closest_hand_vertex: bool = False,
    visualize_mesh_vertex: bool = False,
    visualize_mesh_vertex_label: bool = False,
    visualize_joint: bool = False,
    visualize_joint_label: bool = False,
    visualize_root_position: bool = False,
    current_frame: int = 0,
    fps: int = 20,
    markers_positions: np.ndarray = None,
    visualize_markers: bool = True,
    invalid_point_value: np.ndarray = None,
    calibration_result: np.ndarray = None,
    use_better_optimization: bool = True,
):
    """
    Visualize the hand motion using the SMPL-X model.
    N: number of frames
    M: number of markers

    Parameters:
    - hand_translation (np.ndarray, shape=(N, 3)): Hand translation parameters.
    - hand_orientation (np.ndarray, shape=(N, 3)): Hand orientation parameters, in axis-angle representation.
    - hand_pose (np.ndarray, shape=(N, 45)): Hand pose parameters, in axis-angle representation.
    - hand_shape (np.ndarray, shape=(N, 10)): Hand shape parameters.
    - hand_name (str): The name of the hand. Default is 'hand'.
    - flat_hand_mean (bool): Whether to use the flat hand mean pose. Default is True.
    - visualize_mesh_vertex (bool): Whether to visualize the mesh vertices one by one. Default is False.
    - visualize_mesh_vertex_label (bool): Whether to visualize the mesh vertices index on the mesh vertices. Default is False.
    - visualize_joint (bool): Whether to visualize the joints one by one. Default is False.
    - visualize_joint_label (bool): Whether to visualize the joints index on the joints. Default is False.
    - visualize_root_position (bool): Whether to visualize the root position for each frame. Default is False.
    - current_frame (int): The current first frame index. Default is 0.
    - fps (int): Frames per second for the visualization. Default is 30.
    - markers_positions (np.ndarray, shape=(N, M, 3)): The positions of the markers. Default is None.
    - visualize_markers (bool): Whether to visualize the markers. Default is True.
    - invalid_point_value (np.ndarray, shape=(1, 3)): The value of the invalid points. Default is None.
    - calibration_result (np.ndarray, shape=(N, M)): The calibration result for each marker. Default is None.
    - use_better_optimization (bool): Whether to use better optimization. Default is True.
    """

    batch_size = hand_translation.shape[0]

    # Visualize the motion
    smplx_model = smplx.create(
        model_path=config.data["smpl_model_path"],
        model_type="mano",
        flat_hand_mean=flat_hand_mean,
        is_rhand=False,
        use_pca=False,
        batch_size=batch_size,
    )

    hand_parms = {
        "global_orient": torch.tensor(hand_orientation, dtype=torch.float32),
        "transl": torch.tensor(hand_translation, dtype=torch.float32),
        "hand_pose": torch.tensor(hand_pose, dtype=torch.float32),
        "betas": torch.tensor(hand_shape, dtype=torch.float32),
    }
    smplx_output = smplx_model(**hand_parms)

    root_positions = []

    for frame in range(current_frame, current_frame + batch_size):
        vertices = smplx_output.vertices[frame - current_frame].detach().cpu().numpy()
        joints = smplx_output.joints[frame - current_frame].detach().cpu().numpy()

        # Record root position
        root_positions.append(joints[0])

        rr.set_time("stable_time", duration=frame / fps)  # Keep original timing
        # visualize mesh
        rr.log(
            f"{hand_name}/mesh",
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=smplx_model.faces,
                vertex_normals=compute_vertex_normals(vertices, smplx_model.faces),
            ),
        )

        # visualize markers
        if markers_positions is not None and visualize_markers:
            this_frame_markers_positions = markers_positions[frame - current_frame]
            for i in range(this_frame_markers_positions.shape[0]):
                if np.isclose(
                    this_frame_markers_positions[i], invalid_point_value
                ).all():
                    rr.log(
                        f"marker/point/marker_{i}",
                        entity=rr.Points3D(
                            positions=[[0, 0, 0]],
                            colors=np.full(
                                (1, 3), fill_value=[255, 255, 255], dtype=np.uint8
                            ),
                        ),
                    )
                    if use_better_optimization:
                        rr.log(
                            f"marker/distance/mocap_{i}",
                            entity=rr.LineStrips3D(
                                strips=[[0, 0, 0], [0, 0, 0]],
                                colors=np.full(
                                    (1, 3), fill_value=[255, 255, 255], dtype=np.uint8
                                ),
                            ),
                        )
                    continue
                rr.log(
                    f"marker/point/marker_{i}",
                    entity=rr.Points3D(
                        positions=this_frame_markers_positions[i],
                        radii=0.001,
                        colors=np.full((1, 3), fill_value=[255, 0, 0], dtype=np.uint8),
                    ),
                )

                if visualize_closest_hand_vertex:  # Visualize the distance between the marker and the closest hand vertex
                    # Get vertices of the closest triangle
                    face_vertices = vertices[calibration_result[0][i].astype(int)]

                    # Calculate the closest point using barycentric coordinates
                    closest_point = (
                        face_vertices[0] * calibration_result[1][i][0]
                        + face_vertices[1] * calibration_result[1][i][1]
                        + face_vertices[2] * calibration_result[1][i][2]
                    )

                    # Ensure both points are 1D arrays
                    closest_point = np.array(closest_point).flatten()
                    marker_pos = np.array(this_frame_markers_positions[i]).flatten()

                    rr.log(
                        f"marker/distance/mocap_{i}",
                        entity=rr.LineStrips3D(
                            strips=[[closest_point, marker_pos]],
                            colors=np.full(
                                (1, 3), fill_value=[0, 0, 255], dtype=np.uint8
                            ),
                        ),
                    )

        # visualize joints
        if visualize_joint:
            for i in range(joints.shape[0]):
                rr.log(
                    f"{hand_name}/joint/joint_{i}",
                    rr.Points3D(
                        positions=joints[i],
                        radii=0.001,
                        colors=np.full(joints[i].shape, [0, 255, 0], dtype=np.uint8),
                        labels=[f"{i}"] if visualize_joint_label else None,
                    ),
                )

        # # Parent array for MANO 16 joints (0: wrist, 1-3: Thumb MCP,PIP,DIP, 4-6: Index ..., etc.)
        # # Joint i's parent is parent_array[i]. Root (0) is its own parent.
        # parent_array = np.array(
        #     [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14], dtype=np.int32
        # )
        # # print joint parent distances
        # if joints.shape[0] != len(parent_array):
        #     print(
        #         f"Warning: Frame {frame}: Number of joints ({joints.shape[0]}) does not match parent_array length ({len(parent_array)}). Skipping joint-parent distance calculation."
        #     )
        # else:
        #     print(f"--- Frame {frame} --- Hand: {hand_name} ---")
        #     for i in range(1, joints.shape[0]):  # Start from 1, as joint 0 is the root
        #         parent_idx = parent_array[i]
        #         current_joint_pos = joints[i]
        #         parent_joint_pos = joints[parent_idx]
        #         distance = np.linalg.norm(current_joint_pos - parent_joint_pos)
        #         print(f"  Joint {i} to Parent Joint {parent_idx}: {distance:.4f} m")

        # visualize mesh vertices index
        if visualize_mesh_vertex:
            for i in range(vertices.shape[0]):
                rr.log(
                    f"{hand_name}/mesh_vertex/vertex_{i}",
                    rr.Points3D(
                        positions=vertices[i],
                        radii=0.001,
                        colors=np.full(vertices[i].shape, [255, 0, 0], dtype=np.uint8)
                        if i in vertice_of_patch[0]
                        else np.full(vertices[i].shape, [0, 255, 0], dtype=np.uint8)
                        if i in vertice_of_patch[1]
                        else np.full(vertices[i].shape, [0, 0, 255], dtype=np.uint8)
                        if i in vertice_of_patch[2]
                        else np.full(vertices[i].shape, [255, 255, 0], dtype=np.uint8)
                        if i in vertice_of_patch[3]
                        else np.full(vertices[i].shape, [255, 0, 255], dtype=np.uint8)
                        if i in vertice_of_patch[4]
                        else np.full(vertices[i].shape, [0, 255, 255], dtype=np.uint8)
                        if i in vertice_of_patch[5]
                        else np.full(vertices[i].shape, [128, 0, 0], dtype=np.uint8)
                        if i in vertice_of_patch[6]
                        else np.full(vertices[i].shape, [0, 128, 0], dtype=np.uint8)
                        if i in vertice_of_patch[7]
                        else np.full(vertices[i].shape, [0, 0, 128], dtype=np.uint8)
                        if i in vertice_of_patch[8]
                        else np.full(vertices[i].shape, [128, 128, 0], dtype=np.uint8)
                        if i in vertice_of_patch[9]
                        else np.full(vertices[i].shape, [128, 0, 128], dtype=np.uint8)
                        if i in vertice_of_patch[10]
                        else np.full(vertices[i].shape, [0, 128, 128], dtype=np.uint8)
                        if i in vertice_of_patch[11]
                        else np.full(vertices[i].shape, [192, 192, 192], dtype=np.uint8)
                        if i in vertice_of_patch[12]
                        else np.full(vertices[i].shape, [128, 128, 128], dtype=np.uint8)
                        if i in vertice_of_patch[13]
                        else np.full(vertices[i].shape, [64, 0, 0], dtype=np.uint8)
                        if i in vertice_of_patch[14]
                        else np.full(vertices[i].shape, [0, 64, 0], dtype=np.uint8)
                        if i in vertice_of_patch[15]
                        else np.full(
                            vertices[i].shape, [255, 255, 255], dtype=np.uint8
                        ),
                        labels=[f"{i}"] if visualize_mesh_vertex_label else None,
                    ),
                )

    # Visualize root position for each frame
    if visualize_root_position:
        # Convert root positions to numpy array for visualization
        root_positions = np.array(root_positions)

        for i in range(root_positions.shape[0]):
            rr.set_time("stable_time", duration=i / fps)
            rr.log(
                f"{hand_name}/trajectory",
                rr.Points3D(
                    positions=root_positions[i],
                    radii=0.001,
                    colors=np.full(
                        root_positions[i].shape, [0, 0, 255], dtype=np.uint8
                    ),
                ),
            )


def main():
    raise NotImplementedError("This function is deprecated.")
    init_rerun("MANO_rotation", save=False)

    # Define joint indices
    ROOT_index = 0
    MCP_index = np.array([13, 1, 4, 10, 7])
    PIP_index = np.array([14, 2, 5, 11, 8])
    DIP_index = np.array([15, 3, 6, 12, 9])

    # 1. First hand - neutral pose with local coordinate systems
    hand_translation1 = np.zeros((1, 3))
    hand_orientation1 = np.zeros((1, 3))
    hand_pose1 = np.zeros((1, 45))
    hand_shape1 = np.zeros((1, 10))

    # Get MANO model and joints for neutral pose
    smplx_model = smplx.create(
        model_path=config.data["smpl_model_path"],
        model_type="mano",
        flat_hand_mean=True,
        is_rhand=False,
        use_pca=False,
        batch_size=1,
    )

    hand_parms1 = {
        "global_orient": torch.tensor(hand_orientation1, dtype=torch.float32),
        "transl": torch.tensor(hand_translation1, dtype=torch.float32),
        "hand_pose": torch.tensor(hand_pose1, dtype=torch.float32),
        "betas": torch.tensor(hand_shape1, dtype=torch.float32),
    }
    smplx_output1 = smplx_model(**hand_parms1)
    joints1 = smplx_output1.joints[0].detach().cpu().numpy()

    # Calculate bone directions for neutral pose
    MCP_bone = joints1[MCP_index] - joints1[ROOT_index]
    MCP_bone = MCP_bone / np.linalg.norm(MCP_bone, axis=1, keepdims=True)
    PIP_bone = joints1[PIP_index] - joints1[MCP_index]
    PIP_bone = PIP_bone / np.linalg.norm(PIP_bone, axis=1, keepdims=True)
    DIP_bone = joints1[DIP_index] - joints1[PIP_index]
    DIP_bone = DIP_bone / np.linalg.norm(DIP_bone, axis=1, keepdims=True)

    # Calculate palm normals
    palm_n = np.cross(MCP_bone[:-1], MCP_bone[1:], axis=1)
    palm_n = palm_n / np.linalg.norm(palm_n, axis=1, keepdims=True)

    # Initialize local coordinate systems
    local_coords = np.zeros((16, 3, 3))  # 16 joints, 3 axes (x,y,z)
    local_coords[0] = np.eye(3)  # Root joint

    # Store relative rotations between parent and child joints
    relative_rotations = np.zeros((16, 3, 3))
    relative_rotations[0] = np.eye(3)

    # Set up coordinate systems and calculate relative rotations
    for i, (mcp, pip, dip) in enumerate(zip(MCP_index, PIP_index, DIP_index)):
        # MCP joints
        local_coords[mcp, :, 0] = PIP_bone[i]
        if i == 0 or i == 1:
            local_coords[mcp, :, 1] = palm_n[i]
        elif i == 4:
            local_coords[mcp, :, 1] = palm_n[3]
        else:
            palm_n_i = (palm_n[i - 1] + palm_n[i]) / 2
            local_coords[mcp, :, 1] = palm_n_i

        # Perform Gram-Schmidt orthogonalization on the MCP joint's x and y axes
        local_coords[mcp, :, 0] = local_coords[mcp, :, 0] / np.linalg.norm(
            local_coords[mcp, :, 0]
        )
        local_coords[mcp, :, 1] = (
            local_coords[mcp, :, 1]
            - np.dot(local_coords[mcp, :, 1], local_coords[mcp, :, 0])
            * local_coords[mcp, :, 0]
        )
        local_coords[mcp, :, 1] = local_coords[mcp, :, 1] / np.linalg.norm(
            local_coords[mcp, :, 1]
        )
        local_coords[mcp, :, 2] = np.cross(
            local_coords[mcp, :, 0], local_coords[mcp, :, 1]
        )
        relative_rotations[mcp] = local_coords[0].T @ local_coords[mcp]

        # PIP joints
        local_coords[pip, :, 0] = DIP_bone[i]
        if i == 0 or i == 1:
            local_coords[pip, :, 1] = palm_n[i]
        elif i == 4:
            local_coords[pip, :, 1] = palm_n[3]
        else:
            palm_n_i = (palm_n[i - 1] + palm_n[i]) / 2
            local_coords[pip, :, 1] = palm_n_i
        local_coords[pip, :, 0] = local_coords[pip, :, 0] / np.linalg.norm(
            local_coords[pip, :, 0]
        )
        local_coords[pip, :, 1] = (
            local_coords[pip, :, 1]
            - np.dot(local_coords[pip, :, 1], local_coords[pip, :, 0])
            * local_coords[pip, :, 0]
        )
        local_coords[pip, :, 1] = local_coords[pip, :, 1] / np.linalg.norm(
            local_coords[pip, :, 1]
        )
        local_coords[pip, :, 2] = np.cross(
            local_coords[pip, :, 0], local_coords[pip, :, 1]
        )
        relative_rotations[pip] = local_coords[mcp].T @ local_coords[pip]

        # DIP joints
        local_coords[dip, :, 0] = DIP_bone[i]
        if i == 0 or i == 1:
            local_coords[dip, :, 1] = palm_n[i]
        elif i == 4:
            local_coords[dip, :, 1] = palm_n[3]
        else:
            palm_n_i = (palm_n[i - 1] + palm_n[i]) / 2
            local_coords[dip, :, 1] = palm_n_i
        local_coords[dip, :, 0] = local_coords[dip, :, 0] / np.linalg.norm(
            local_coords[dip, :, 0]
        )
        local_coords[dip, :, 1] = (
            local_coords[dip, :, 1]
            - np.dot(local_coords[dip, :, 1], local_coords[dip, :, 0])
            * local_coords[dip, :, 0]
        )
        local_coords[dip, :, 1] = local_coords[dip, :, 1] / np.linalg.norm(
            local_coords[dip, :, 1]
        )
        local_coords[dip, :, 2] = np.cross(
            local_coords[dip, :, 0], local_coords[dip, :, 1]
        )
        relative_rotations[dip] = local_coords[pip].T @ local_coords[dip]

    def is_orthogonal(matrix):
        """
        Check if a matrix is orthogonal.

        Parameters:
        - matrix (np.ndarray, shape=(3, 3)): Matrix to check.

        Returns:
        - bool: True if the matrix is orthogonal, False otherwise.
        """
        identity = np.eye(matrix.shape[0])
        return np.allclose(matrix.T @ matrix, identity) and np.allclose(
            matrix @ matrix.T, identity
        )

    # Check if local coordinate systems are orthogonal
    for i in range(16):
        if not is_orthogonal(local_coords[i]):
            raise ValueError(f"Local coordinate system at joint {i} is not orthogonal.")

    # Visualize first hand with coordinate systems
    visualize_hand(
        hand_translation=hand_translation1,
        hand_orientation=hand_orientation1,
        hand_pose=hand_pose1,
        hand_shape=hand_shape1,
        hand_name="hand1",
        visualize_joint=True,
        visualize_joint_label=True,
    )

    # Draw coordinate systems for neutral pose
    for i in range(16):
        rr.log(
            f"hand1/coords/joint_{i}",
            rr.Arrows3D(
                origins=np.tile(joints1[i], (3, 1)),
                vectors=local_coords[i].T * 0.02,
                radii=0.001,
                colors=np.array(
                    [[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8
                ),
            ),
        )

    def dof_to_rot_vector_legacy(dof, dof_value):
        assert np.sum(dof) == dof_value.shape[0]
        parent_index = np.array([0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14])
        rot_vector = np.zeros((16, 3))

        # Calculate local coordinate system
        local_coords = np.zeros((16, 3, 3))
        local_coords[0] = np.eye(3)
        local_coords_after_rotation = np.zeros((16, 3, 3))
        local_coords_after_rotation[0] = np.eye(3)
        joint_orientations = np.zeros((16, 3, 3))
        joint_orientations[0] = np.eye(3)

        value_idx = 0
        for i in range(16):
            parent = parent_index[i]

            # Apply rotations based on DOF
            if dof[i] == 1:
                # Only Z rotation
                z_angle = dof_value[value_idx]
                rot_mat = R.from_euler("Z", z_angle, degrees=False).as_matrix()
                value_idx += 1
            elif dof[i] == 2:
                # Y and Z rotations
                y_angle = dof_value[value_idx]
                z_angle = dof_value[value_idx + 1]
                rot_mat = R.from_euler(
                    "YZ", [y_angle, z_angle], degrees=False
                ).as_matrix()
                value_idx += 2
            elif dof[i] == 3:
                # X, Y, and Z rotations
                x_angle = dof_value[value_idx]
                y_angle = dof_value[value_idx + 1]
                z_angle = dof_value[value_idx + 2]
                rot_mat = R.from_euler(
                    "XYZ", [x_angle, y_angle, z_angle], degrees=False
                ).as_matrix()
                value_idx += 3

            # Update accumulated rotation
            local_coords[i] = (
                local_coords_after_rotation[parent] @ relative_rotations[i]
            )
            local_coords_after_rotation[i] = local_coords[i] @ rot_mat
            joint_orientations[i] = joint_orientations[parent] @ rot_mat

            # Convert to axis-angle representation
            rot_vector[i] = R.from_matrix(
                joint_orientations[parent].T
                @ (local_coords[i] @ rot_mat @ local_coords[i].T)
                @ joint_orientations[parent]
            ).as_rotvec()

        return rot_vector, local_coords

    # Test with some example DOF values
    dof_with_root = np.array([3, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1])
    dof_value = np.zeros(24)
    dof_value = [
        [-np.pi / 6, 0, 0],
        [-np.pi / 3, 0],
        [-np.pi / 2],
        [-np.pi / 2],
        [0, 0],
        [-np.pi / 2],
        [np.pi / 2],
        [np.pi / 6, 0],
        [0],
        [-np.pi / 2],
        [0, 0],
        [0],
        [0],
        [-np.pi / 6, 0],
        [np.pi / 3, 0],
        [-np.pi / 2],
    ]
    dof_value = np.concatenate(dof_value, axis=0)
    rot_vectors, local_coords = dof_to_rot_vector_legacy(dof_with_root, dof_value)

    # Convert rotation vectors to MANO format (45D vector)
    hand_translation2 = np.array([[0, 0.1, 0]])  # Offset from first hand
    hand_orientation2 = rot_vectors[0][None, :]  # Root joint rotation
    hand_pose2 = rot_vectors[1:].reshape(1, -1)  # Exclude root joint

    # Visualize second hand with coordinate systems
    visualize_hand(
        hand_translation=hand_translation2,
        hand_orientation=hand_orientation2,
        hand_pose=hand_pose2,
        hand_shape=hand_shape1,  # Use same shape as first hand
        hand_name="hand2",
        visualize_joint=True,
        visualize_joint_label=True,
    )

    # Get joints for second hand
    hand_parms2 = {
        "global_orient": torch.tensor(hand_orientation2, dtype=torch.float32),
        "transl": torch.tensor(hand_translation2, dtype=torch.float32),
        "hand_pose": torch.tensor(hand_pose2, dtype=torch.float32),
        "betas": torch.tensor(hand_shape1, dtype=torch.float32),
    }
    smplx_output2 = smplx_model(**hand_parms2)
    joints2 = smplx_output2.joints[0].detach().cpu().numpy()

    # Draw coordinate systems for rotated hand
    for i in range(16):
        rr.log(
            f"hand2/coords/joint_{i}",
            rr.Arrows3D(
                origins=np.tile(joints2[i], (3, 1)),
                vectors=local_coords[i].T * 0.02,
                radii=0.001,
                colors=np.array(
                    [[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8
                ),
            ),
        )


if __name__ == "__main__":
    main()
