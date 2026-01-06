import numpy as np
import smplx
import torch
import torch.optim as optim
from tqdm import tqdm

import wandb
from HandReconstruction import config
from HandReconstruction.Data.mano_segment import vertice_of_patch
from HandReconstruction.Loss.loss_regularization import loss_regularization_dof
from HandReconstruction.Loss.loss_vertex import (
    loss_vertex_meshdis_calibration,
    loss_vertex_meshdis_inference,
)
from HandReconstruction.Utility.utils_hand import dof_to_rot_vector, index_dof, num_dof
from HandReconstruction.Utility.utils_pytorch3d import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

# Global variables for tracking loss across frames
# meshdis calibration: find the nearest point instead of nearest vertex
# a: shape = (num_valid_points, 3 vertex indices of one face): one valid point corresponds to one face of the mesh, all 0 if not calibrated
# b: shape = (num_valid_points, 3 barycentric coordinates): one valid point corresponds to one face of the mesh, all 0 if not calibrated
# calibration_result = [a, b]

calibration_result = None
global_vertex_loss = []
global_regularization_loss = []


def setup_optimizer(
    hand_translation,
    hand_orientation_6d,
    hand_dof,
    root_valid_mocap_point_count=0,
):
    """
    Set up the optimizer based on the number of valid mocap points for the root.

    Args:
        hand_translation: Hand translation parameter tensor
        hand_orientation_6d: Hand orientation parameter tensor (6D representation)
        hand_dof: Hand degrees of freedom parameter tensor
        root_valid_mocap_point_count: Number of valid mocap points for the root

    Returns:
        The configured optimizer
    """
    if (
        root_valid_mocap_point_count <= config.optimize["min_marker_for_calibration"]
    ):  # if all markers for the root are missing
        return optim.Adam(
            [hand_translation, hand_orientation_6d], lr=config.optimize["learning_rate"]
        )
    else:
        return optim.Adam(
            [hand_translation, hand_orientation_6d, hand_dof],
            lr=config.optimize["learning_rate"],
        )


def calculate_vertex_loss(
    vertices,
    vertice_of_joint_device,
    hand_face,
    target_points,
    target_indices,
    valid_mask,
    calibration_result,
    is_calibration,
    device,
):
    """
    Calculate the vertex loss for all joints.

    Args:
        vertices (torch.Tensor, shape=(num_vertices, 3)): The MANO model vertices in world coordinate
        vertice_of_joint_device (list of torch.Tensor): List of vertex indices for each joint segment
        hand_face (torch.Tensor, shape=(num_faces, 3)): Face indices defining mesh triangles
        target_points (torch.Tensor, shape=(num_points, 3)): Target mocap points in world coordinate
        target_indices (torch.Tensor, shape=(num_points,)): Joint segment index for each target point
        valid_mask (torch.Tensor, shape=(num_points,)): Boolean mask indicating valid target points
        calibration_result ([np.ndarray, np.ndarray], shape=[(num_points, 3), (num_points, 3)]): Current calibration result containing barycentric indices and coordinates
        is_calibration (bool): Whether this is a calibration run
        device (torch.device): Device to run calculations on

    Returns:
        tuple:
            - vertex_loss (torch.Tensor, shape=(1,)): Weighted average vertex loss across all segments
            - updated_calibration_result (list of np.ndarray): Updated calibration result if in calibration mode
            - total_weight (float): Sum of weights used for loss normalization
    """
    vertex_loss = torch.tensor(0, dtype=torch.float32, device=device)
    total_weight = 0  # Calculate the total weight, used to calculate the average loss

    # Process each joint segment
    for i in range(config.hand["total_segment_num"]):
        mocap_point_mask = (
            target_indices == i
        ) & valid_mask  # Index of all the mocap points
        mocap_point_index = torch.where(mocap_point_mask)[0]

        if mocap_point_mask.sum() == 0:
            continue

        # Calculate loss and update total weight
        if is_calibration:  # Calibration phase, use part-2-part correspondence
            updated_vertex_loss, updated_calibration_result = (
                loss_vertex_meshdis_calibration(
                    vertices,
                    vertice_of_joint_device[i],
                    hand_face,
                    target_points,
                    mocap_point_index,
                    calibration_result,
                )
            )
            calibration_result = updated_calibration_result
        else:  # Inference phase, use point-to-point correspondence
            updated_vertex_loss = loss_vertex_meshdis_inference(
                vertices,
                vertice_of_joint_device[i],
                target_points,
                mocap_point_mask,
                calibration_result,
            )

        # Apply weighting for end effectors
        weight = (
            config.optimize["weight_for_end_effector"]
            if i in config.optimize["end_effector_index"]
            else 1
        )
        total_weight += weight
        vertex_loss += updated_vertex_loss * weight / torch.sum(mocap_point_mask)

    # Average the loss over all weighted points
    if total_weight > 0:
        vertex_loss /= total_weight

    return vertex_loss, calibration_result, total_weight


def process_invalid_markers(hand_dof, init_hand_dof, invalid_marker_indices, device):
    """
    Process invalid markers by replacing their rotations with initial pose.

    Iterates through fingers from tip to base. If a marker segment is invalid,
    its corresponding DoF is reset to the initial pose. If a segment's marker
    is valid, the processing for that finger stops, even if segments closer
    to the base have invalid markers.

    Args:
        hand_dof (torch.Tensor, shape=(1, num_dof)): Current hand DOF parameters.
        init_hand_dof (torch.Tensor, shape=(1, num_dof)): Initial hand DOF parameters.
        invalid_marker_indices (list[int]): Indices of invalid markers, 0-16, including root.
        device (torch.device): Device to run calculations on.

    Returns:
        torch.Tensor, shape=(1, num_dof): The updated hand_dof tensor.
    """
    hand_dof_updated = hand_dof.clone().detach().to(device)
    invalid_marker_indices_set = set(invalid_marker_indices)  # Use set for O(1) lookups
    use_previous_dof = []

    # Define finger segments, ordered from tip marker index to base marker index
    # Based on the original logic: Index(3,2,1), Middle(6,5,4), Ring(12,11,10), Pinky(9,8,7), Thumb(15,14,13)
    finger_marker_indices_map = [
        [3, 2, 1],  # Index
        [6, 5, 4],  # Middle
        [12, 11, 10],  # Ring
        [9, 8, 7],  # Pinky
        [15, 14, 13],  # Thumb
    ]

    for finger_segments in finger_marker_indices_map:
        # Iterate from tip to base for each finger
        for marker_index in finger_segments:
            if marker_index in invalid_marker_indices_set:
                dof_index = marker_index - 1
                # If the marker is invalid, reset its DoF to the initial pose
                start_idx = index_dof[dof_index]
                num_finger_dof = config.hand["dof"][dof_index]
                end_idx = start_idx + num_finger_dof
                hand_dof_updated[0, start_idx:end_idx] = init_hand_dof[
                    0, start_idx:end_idx
                ]
                # print(f"Reset DoF for marker {marker_index}, start_idx: {start_idx}, end_idx: {end_idx}")
                # print(f"previous hand_dof: {hand_dof[0, start_idx:end_idx]}")
                # print(f"init_hand_dof: {init_hand_dof[0, start_idx:end_idx]}")
                use_previous_dof.append(marker_index)  # Keep track of reset DoFs
            else:
                # If a valid marker is found along the finger, stop processing this finger
                # as per the original logic's requirement.
                break

    # print(f"use_previous_dof: {sorted(list(set(use_previous_dof)))}") # Print unique sorted indices
    return hand_dof_updated


def find_invalid_markers(
    target_indices,
    valid_mask,
):
    """
    Find invalid markers and calculate the total weight for loss calculation.

    Args:
        target_indices: Indices of target points
        valid_mask: Mask of valid points

    Returns:
        tuple: (invalid_marker_indices, total_weight, root_valid_mocap_point_count)
    """
    invalid_marker_mask = np.zeros(
        (config.hand["total_joint_num"],)
    )  # The joints that do not exist marker
    total_weight = 0  # Calculate the total weight, used to calculate the average loss

    for i in range(
        config.hand["total_segment_num"]
    ):  # +1 because index 0 is root joint
        mocap_point_mask = (target_indices == i) & valid_mask

        # Check for invalid markers
        if mocap_point_mask.sum() == 0 and i <= 15:  # No valid points for this joint
            invalid_marker_mask[i] = 1
        elif mocap_point_mask.sum() != 0 and i == 16:  # Valid points for wrist joint
            invalid_marker_mask[0] = 0

        # Update total weight for valid markers
        if mocap_point_mask.sum() != 0:
            total_weight += (
                config.optimize["weight_for_end_effector"]
                if i in config.optimize["end_effector_index"]
                else 1
            )

    invalid_marker_indices = np.where(invalid_marker_mask == 1)[0]

    # Count valid mocap points for the root
    root_valid_mocap_point_count = ((target_indices == 0) & valid_mask).sum() + (
        (target_indices == 16) & valid_mask
    ).sum()

    return invalid_marker_indices, total_weight, root_valid_mocap_point_count


def optimize_hand_pose(
    frame_index,
    target_points,
    target_indices,
    init_hand_translation=None,
    init_hand_orientation=None,
    init_hand_dof=None,
    init_hand_pose=None,
    init_hand_shape=None,
    num_iterations=2000,
    is_calibration=False,
):
    """
    Optimize hand pose for a single frame to match the target points.

    Parameters:
    - frame_index (int): Index of the frame to optimize.
    - target_points (np.ndarray, shape=(num_points, 3)): Target points in world coordinate for a single frame.
    - target_indices (np.ndarray, shape=(num_points,)): Indices of the target points of which fingers they belong to.
    - init_hand_translation (np.ndarray, shape=(3,)): Initial hand translation parameters.
    - init_hand_orientation (np.ndarray, shape=(3,)): Initial hand orientation parameters, in axis-angle representation.
    - init_hand_dof (np.ndarray, shape=(num_dof,)): Initial hand DOF parameters.
    - init_hand_pose (np.ndarray, shape=(45,)): Initial hand pose parameters, in axis-angle representation.
    - init_hand_shape (np.ndarray, shape=(10,)): Initial hand shape parameters.
    - num_iterations (int): Number of optimization iterations.
    - is_calibration (bool): Whether this is calibration phase.

    Returns:
    - optimized_hand_translation (np.ndarray, shape=(3,)): Optimized hand translation parameters.
    - optimized_hand_orientation (np.ndarray, shape=(3,)): Optimized hand orientation parameters, in axis-angle representation.
    - optimized_hand_pose (np.ndarray, shape=(45,)): Optimized hand pose parameters, in axis-angle representation.
    - optimized_hand_dof (np.ndarray, shape=(num_dof,)): Optimized hand DOF parameters.
    - optimized_hand_shape (np.ndarray, shape=(10,)): Optimized hand shape parameters.
    """
    global calibration_result, global_vertex_loss, global_regularization_loss

    # Set device
    device = torch.device(config.optimize["device"])

    # Initialize default parameters if not provided
    if init_hand_translation is None:
        init_hand_translation = np.zeros((1, 3))
    if init_hand_orientation is None:
        init_hand_orientation = np.zeros((1, 3))
    if init_hand_dof is None:
        init_hand_dof = np.zeros((1, num_dof))
    if init_hand_pose is None:
        init_hand_pose = np.zeros((1, 45))
    if init_hand_shape is None:
        init_hand_shape = np.zeros((1, 10))

    # Convert to torch tensor and move to device
    init_hand_translation = torch.as_tensor(
        init_hand_translation, dtype=torch.float32, device=device
    ).view(1, 3)
    init_hand_orientation = torch.as_tensor(
        init_hand_orientation, dtype=torch.float32, device=device
    ).view(1, 3)
    init_hand_pose = torch.as_tensor(
        init_hand_pose, dtype=torch.float32, device=device
    ).view(1, 45)
    init_hand_shape = torch.as_tensor(
        init_hand_shape, dtype=torch.float32, device=device
    ).view(1, 10)
    init_hand_dof = torch.as_tensor(
        init_hand_dof, dtype=torch.float32, device=device
    ).view(1, num_dof)

    # Initialize hand parameters for optimization
    hand_translation = init_hand_translation.clone().requires_grad_(True)
    hand_orientation_6d = (
        matrix_to_rotation_6d(axis_angle_to_matrix(init_hand_orientation))
        .clone()
        .requires_grad_(True)
    )
    hand_dof = init_hand_dof.clone().requires_grad_(True)
    hand_shape = init_hand_shape.clone().requires_grad_(True)

    # Create SMPL-X model
    smplx_model = smplx.create(
        model_path=config.data["smpl_model_path"],
        model_type="mano",
        flat_hand_mean=True,
        is_rhand=False,
        use_pca=False,
        batch_size=1,
    ).to(device)
    hand_face = torch.tensor(smplx_model.faces, dtype=torch.int32, device=device)

    # Create a mask to exclude invalid points
    valid_mask = ~(
        np.all(np.isclose(target_points, config.data["invalid_point_value"]), axis=1)
    )
    if np.sum(valid_mask) == 0:
        print(f"Warning: no valid markers for frame {frame_index}, skip this frame!")
        ans = {
            "optimized_hand_translation": init_hand_translation.detach()
            .cpu()
            .numpy()[0],
            "optimized_hand_orientation": init_hand_orientation.detach()
            .cpu()
            .numpy()[0],
            "optimized_hand_dof": init_hand_dof.detach().cpu().numpy()[0],
            "optimized_hand_pose": init_hand_pose.detach().cpu().numpy()[0],
            "optimized_hand_shape": init_hand_shape.detach().cpu().numpy()[0],
        }
        global_vertex_loss.append(np.nan)
        global_regularization_loss.append(np.nan)
        return ans

    # Find invalid markers and calculate weights
    invalid_marker_indices, total_weight, root_valid_mocap_point_count = (
        find_invalid_markers(
            target_indices,
            valid_mask,
        )
    )

    # Set up optimizer
    optimizer = setup_optimizer(
        hand_translation,
        hand_orientation_6d,
        hand_dof,
        root_valid_mocap_point_count,
    )

    # Move data to device
    vertice_of_joint_device = [
        torch.tensor(vertice_of_patch[i], dtype=torch.int32, device=device)
        for i in range(config.hand["total_segment_num"])
    ]
    target_points = torch.tensor(target_points, dtype=torch.float32, device=device)
    target_indices = torch.tensor(target_indices, dtype=torch.int32, device=device)
    valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=device)

    # Optimize
    for iteration in tqdm(
        range(num_iterations),
        desc=f"{is_calibration and 'Calibrating' or 'Optimizing'} hand pose for frame {frame_index}",
    ):
        optimizer.zero_grad()

        # Convert 6D to rotation matrices
        hand_orientation_matrix = rotation_6d_to_matrix(hand_orientation_6d)

        # Set up MANO parameters
        hand_parms = {
            "global_orient": matrix_to_axis_angle(hand_orientation_matrix),
            "transl": hand_translation,
            "hand_pose": dof_to_rot_vector(hand_dof)[0].view(1, 45),
            "betas": hand_shape,
        }

        # Run forward pass
        smplx_output = smplx_model(**hand_parms)
        vertices = smplx_output.vertices[0]

        # Calculate vertex loss
        vertex_loss, updated_calibration_result, _ = calculate_vertex_loss(
            vertices,
            vertice_of_joint_device,
            hand_face,
            target_points,
            target_indices,
            valid_mask,
            calibration_result,
            is_calibration,
            device,
        )

        if is_calibration:
            calibration_result = updated_calibration_result

        if torch.isclose(
            vertex_loss, torch.tensor(0, dtype=torch.float32, device=device)
        ):
            print(
                f"Warning: vertice_loss is 0 for frame {frame_index}, no valid markers for this frame!"
            )

        # Calculate regularization loss and total loss
        regularization_loss = loss_regularization_dof(hand_dof)
        loss = (
            vertex_loss + regularization_loss * config.optimize["regularization_weight"]
        )

        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()

        # Log loss to wandb if not in test mode
        if (
            not config.optimize["test_mode"]
            and frame_index % config.data["save_one_frame_curve_every_n_frames"] == 0
        ):
            wandb.log(
                {
                    "Local_vertex/step": iteration,
                    "Local_regularization/step": iteration,
                    f"Local_vertex/frame_{frame_index:05d}": vertex_loss.item(),
                    f"Local_regularization/frame_{frame_index:05d}": regularization_loss.item()
                    if root_valid_mocap_point_count
                    > config.optimize["min_marker_for_calibration"]
                    else 0,
                }
            )

    # Save the loss values
    global_vertex_loss.append(vertex_loss.item())
    global_regularization_loss.append(regularization_loss.item())

    # Process results
    optimized_hand_orientation = (
        matrix_to_axis_angle(rotation_6d_to_matrix(hand_orientation_6d))
        .detach()
        .cpu()
        .numpy()
    )

    # Handle invalid markers
    hand_dof = process_invalid_markers(
        hand_dof, init_hand_dof, invalid_marker_indices, device
    )

    # Convert final pose to axis-angle
    optimized_hand_pose = dof_to_rot_vector(hand_dof)[0].view(1, 45).detach().cpu().numpy()
    optimized_hand_translation = hand_translation.detach().cpu().numpy()
    optimized_hand_shape = hand_shape.detach().cpu().numpy()

    # Return results
    return {
        "optimized_hand_translation": optimized_hand_translation[0],
        "optimized_hand_orientation": optimized_hand_orientation[0],
        "optimized_hand_dof": hand_dof[0].detach().cpu().numpy(),
        "optimized_hand_pose": optimized_hand_pose[0],
        "optimized_hand_shape": optimized_hand_shape[0],
    }


def get_calibration_result():
    """Get the current calibration result."""
    return calibration_result


def set_calibration_result(new_calibration_result):
    """Set the calibration result."""
    global calibration_result
    calibration_result = new_calibration_result


def get_loss_data():
    """Get the collected loss data."""
    return {
        "vertex_loss": global_vertex_loss,
        "regularization_loss": global_regularization_loss,
    }


def clear_loss_data():
    """Clear the collected loss data."""
    global global_vertex_loss, global_regularization_loss
    global_vertex_loss = []
    global_regularization_loss = []
