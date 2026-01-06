import datetime
import os

import numpy as np
import torch
from tqdm import tqdm

import wandb
from HandReconstruction import config
from HandReconstruction.hand_optimizer import (
    clear_loss_data,
    get_calibration_result,
    get_loss_data,
    optimize_hand_pose,
    set_calibration_result,
)
from HandReconstruction.Utility.utils_hand import num_dof
from HandReconstruction.Utility.utils_visualize import init_rerun, visualize_hand

if __name__ == "__main__":
    # Set devices
    device = torch.device(config.optimize["device"])
    print(f"Using device: {device}")

    # Load data
    with open(config.data["hand_mocap_point_data_path"], "rb") as f:
        points = np.load(f)  # [num_frames, num_points, 3]
        num_total_frame = points.shape[0]
        num_marker = points.shape[1]
        print(f"data: Total frame count: {num_total_frame}, Marker count: {num_marker}")

    with open(config.data["body_part_index_path"], "rb") as f:
        body_part_indices = np.load(f)

    # Read configuration
    fps = config.data["fps"]
    set_calibration_result([np.zeros((num_marker, 3)), np.zeros((num_marker, 3))])
    hand_shape = np.array(config.hand["MANO_shape"])

    # Apply cutoff parameters
    cutoff_start = config.data["cutoff_start"]
    cutoff_end = config.data["cutoff_end"]
    if cutoff_end == -1:
        cutoff_end = num_total_frame
    cutoff_step = config.data["cutoff_step"]
    points = points[cutoff_start:cutoff_end:cutoff_step]
    num_total_frame = points.shape[0]
    num_marker = points.shape[1]
    fps /= cutoff_step

    # Generate timestamp for this run
    time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Initialize wandb if not in test mode
    if not config.optimize["test_mode"]:
        wandb.init(
            project="hand_mocap",
            name=f"{config.mocap_session_name}_{time_str}",
            config={
                "num_total_frame": num_total_frame,
                "num_marker": num_marker,
                "current_running_frame": 0,
                "cutoff_start": cutoff_start,
                "cutoff_end": cutoff_end,
                "cutoff_step": cutoff_step,
                "fps": fps,
                "num_frame_for_calibration": config.optimize[
                    "num_frame_for_calibration"
                ],
                "num_iterations": config.optimize["num_iterations"],
                "learning_rate": config.optimize["learning_rate"],
                "regularization_weight": config.optimize["regularization_weight"],
                "end_effector_index": config.optimize["end_effector_index"],
                "weight_for_end_effector": config.optimize["weight_for_end_effector"],
                "test_mode": config.optimize["test_mode"],
                "use_shape_fitting": config.optimize["use_shape_fitting"],
                "num_iterations_init_frame": config.optimize[
                    "num_iterations_init_frame"
                ],
                "index_init_frame": config.optimize["index_init_frame"],
                "total_joint_num": config.hand["total_joint_num"],
                "total_segment_num": config.hand["total_segment_num"],
                "dof": config.hand["dof"],
            },
        )

        # Define wandb metrics
        wandb.define_metric("Local_vertex/step")
        wandb.define_metric("Local_vertex/*", step_metric="Local_vertex/step")
        wandb.define_metric("Local_regularization/step")
        wandb.define_metric(
            "Local_regularization/*", step_metric="Local_regularization/step"
        )
        wandb.define_metric("Global/step")
        wandb.define_metric("Global/*", step_metric="Global/step")
        wandb.define_metric("Metric/step")
        wandb.define_metric("Metric/*", step_metric="Metric/step")

    # Initialize result arrays
    optimized_translations = np.zeros((num_total_frame, 3))
    optimized_orientations = np.zeros((num_total_frame, 3))
    optimized_dofs = np.zeros((num_total_frame, num_dof))
    optimized_poses = np.zeros((num_total_frame, 45))
    optimized_shapes = np.zeros((num_total_frame, 10))

    # Clear any previous loss data
    clear_loss_data()

    # Process each frame
    for frame_cnt in tqdm(range(num_total_frame), desc="Optimizing hand pose"):
        if not config.optimize["test_mode"]:
            wandb.config.update(
                {"current_running_frame": frame_cnt}, allow_val_change=True
            )

        # Initialize rerun session for visualization
        if frame_cnt % config.data["save_rerun_and_MANO_every_n_frames"] == 0:
            rerun_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Result",
                "RerunSession",
                config.mocap_session_name,
                time_str,
                f"frame_{frame_cnt:05d}_{min(frame_cnt + config.data['save_rerun_and_MANO_every_n_frames'], num_total_frame):05d}.rrd",
            )
            init_rerun(
                rerun_path,
                save=not config.optimize["test_mode"],
            )

        # Get initial parameters for current frame, segment-wise initialization
        init_hand_translation = (
            np.zeros((1, 3))
            if frame_cnt in config.optimize["index_init_frame"]
            else optimized_translations[frame_cnt - 1]
        )
        init_hand_orientation = (
            np.zeros((1, 3))
            if frame_cnt in config.optimize["index_init_frame"]
            else optimized_orientations[frame_cnt - 1]
        )
        init_hand_dof = (
            np.zeros((1, num_dof))
            if frame_cnt in config.optimize["index_init_frame"]
            else optimized_dofs[frame_cnt - 1]
        )
        init_hand_pose = (
            np.zeros((1, 45))
            if frame_cnt in config.optimize["index_init_frame"]
            else optimized_poses[frame_cnt - 1]
        )

        # Select hand shape
        if frame_cnt > 0:
            init_hand_shape = optimized_shapes[frame_cnt - 1]
        else:
            init_hand_shape = (
                hand_shape.reshape(1, 10)
                if config.optimize["use_shape_fitting"]
                else np.zeros((1, 10))
            )

        # Determine if this is a frame needing special handling (e.g., more iterations)
        is_init_frame = frame_cnt in config.optimize["index_init_frame"]
        num_iterations = (
            config.optimize["num_iterations_init_frame"]
            if is_init_frame
            else config.optimize["num_iterations"]
        )

        # Determine if this is a calibration frame
        is_calibration = frame_cnt < config.optimize["num_frame_for_calibration"]

        # Optimize hand pose for current frame
        hand_motion = optimize_hand_pose(
            frame_index=frame_cnt * cutoff_step + cutoff_start,
            target_points=points[frame_cnt],
            target_indices=body_part_indices,
            init_hand_translation=init_hand_translation,
            init_hand_orientation=init_hand_orientation,
            init_hand_dof=init_hand_dof,
            init_hand_pose=init_hand_pose,
            init_hand_shape=init_hand_shape,
            num_iterations=num_iterations,
            is_calibration=is_calibration,
        )

        # Store optimization results
        optimized_translations[frame_cnt] = hand_motion["optimized_hand_translation"]
        optimized_orientations[frame_cnt] = hand_motion["optimized_hand_orientation"]
        optimized_dofs[frame_cnt] = hand_motion["optimized_hand_dof"]
        optimized_poses[frame_cnt] = hand_motion["optimized_hand_pose"]
        optimized_shapes[frame_cnt] = hand_motion["optimized_hand_shape"]

        # Visualize optimized hand
        visualize_hand(
            hand_translation=np.array(optimized_translations[frame_cnt].reshape(1, 3)),
            hand_orientation=np.array(optimized_orientations[frame_cnt].reshape(1, 3)),
            hand_pose=np.array(optimized_poses[frame_cnt].reshape(1, 45)),
            hand_shape=np.array(optimized_shapes[frame_cnt].reshape(1, 10)),
            current_frame=frame_cnt * cutoff_step + cutoff_start,
            fps=fps,
            markers_positions=points[frame_cnt].reshape(1, num_marker, 3),
            visualize_markers=True,
            invalid_point_value=config.data["invalid_point_value"],
            calibration_result=get_calibration_result(),
            visualize_closest_hand_vertex=True,
            visualize_mesh_vertex=False,
            visualize_mesh_vertex_label=False,
        )

        loss_data = get_loss_data()
        vertex_loss = (
            loss_data["vertex_loss"][frame_cnt]
            if frame_cnt < len(loss_data["vertex_loss"])
            else None
        )
        regularization_loss = (
            loss_data["regularization_loss"][frame_cnt]
            if frame_cnt < len(loss_data["regularization_loss"])
            else None
        )

        # Log loss values to wandb
        if not config.optimize["test_mode"]:
            if vertex_loss is not None and regularization_loss is not None:
                wandb.log(
                    {
                        "Global/step": frame_cnt * cutoff_step + cutoff_start,
                        "Global/Vertex": vertex_loss,
                        "Global/Regularization": regularization_loss,
                    }
                )
        else:
            print(
                f"\nFrame {frame_cnt}: Vertex loss {vertex_loss}, Regularization loss {regularization_loss}\n"
            )

        # Save intermediate results
        if not config.optimize["test_mode"] and (
            (frame_cnt + 1) % config.data["save_rerun_and_MANO_every_n_frames"] == 0
            or (frame_cnt == num_total_frame - 1)
        ):
            # Calculate range of frames to save
            start_frame = (
                (frame_cnt * cutoff_step + cutoff_start)
                // config.data["save_rerun_and_MANO_every_n_frames"]
            ) * config.data["save_rerun_and_MANO_every_n_frames"]
            end_frame = min(
                start_frame + config.data["save_rerun_and_MANO_every_n_frames"],
                num_total_frame,
            )

            # Create output directory if it doesn't exist
            optimized_data_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Result",
                config.mocap_session_name,
                time_str,
                f"frame_{start_frame:05d}_{end_frame:05d}.npz",
            )
            folder_path = os.path.dirname(optimized_data_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Save the data
            np.savez(
                optimized_data_path,
                optimized_dofs=optimized_dofs[start_frame:end_frame],
                optimized_poses=optimized_poses[start_frame:end_frame],
                optimized_shapes=optimized_shapes[start_frame:end_frame],
                optimized_translations=optimized_translations[start_frame:end_frame],
                optimized_orientations=optimized_orientations[start_frame:end_frame],
            )

    # Calculate and log final metrics
    loss_data = get_loss_data()
    vertex_loss = np.array(loss_data["vertex_loss"])
    regularization_loss = np.array(loss_data["regularization_loss"])

    # Log final metrics to wandb
    if not config.optimize["test_mode"]:
        wandb.log(
            {
                "Metric/step": 0,
                "Metric/Vertex": np.nanmean(vertex_loss),
                "Metric/Regularization": np.nanmean(regularization_loss),
                "Metric/Loss_Total": np.nanmean(
                    vertex_loss
                    + regularization_loss * config.optimize["regularization_weight"]
                ),
            }
        )
        wandb.finish()
