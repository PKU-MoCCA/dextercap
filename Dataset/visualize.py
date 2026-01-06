import argparse
import math
import os

import numpy as np
import rerun as rr
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from ObjectReconstruction.rubikscube import visualize_rubiks_cube_animation

MANO_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    os.path.pardir,
    "HandReconstruction",
    "Data",
    "HumanModels",
)


def compute_vertex_normals(vertices, faces):
    """
    Calculate vertex normals using vectorized operations.

    Parameters:
    - vertices (np.ndarray, shape=(N, 3)): Array of vertex coordinates.
    - faces (np.ndarray, shape=(M, 3)): Array of vertex indices for each face.

    Returns:
    - np.ndarray, shape=(N, 3): Array of normalized vertex normals.
    """
    # Get the vertices of the triangles
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute the normal vectors for each face
    normals = np.cross(v1 - v0, v2 - v0)

    # Compute the lengths of the normal vectors
    norm_lengths = np.linalg.norm(normals, axis=1)

    # Avoid division by zero, set the normal vectors with zero length to a small value
    norm_lengths[norm_lengths == 0] = 1e-10

    # Normalize the normal vectors
    normals /= norm_lengths[:, np.newaxis]

    # Add the normal vectors to the vertices
    vertex_normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], normals)

    # Compute the lengths of the vertex normals
    vertex_norm_lengths = np.linalg.norm(vertex_normals, axis=1)

    # Avoid division by zero, set the normal vectors with zero length to a small value
    vertex_norm_lengths[vertex_norm_lengths == 0] = 1e-10

    # Normalize the vertex normals
    vertex_normals = (vertex_normals.T / vertex_norm_lengths).T
    return vertex_normals


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
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)  # Set an up-axis = -Z
    # rr.log("", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # Set an up-axis = +Y
    rr.set_time("stable_time", duration=0)


global_cylinder_resolution = -1
global_cylinder_vertices = []
global_cylinder_indices = []


def visualize_cylinder(
    radius=0.0275,
    height=0.16,
    resolution=128,
    translation=np.zeros(3),
    orientation=np.eye(3),
    color=[0, 255, 0],
):
    global global_cylinder_resolution
    global global_cylinder_vertices
    global global_cylinder_indices

    if global_cylinder_resolution == resolution:
        vertices = global_cylinder_vertices
        indices = global_cylinder_indices

        R_object = R.from_quat(orientation)
        vertices = R_object.apply(vertices)
        vertices = vertices + translation

        # Render the cylinder
        rr.log(
            "object/cylinder",
            rr.Mesh3D(
                vertex_positions=vertices,
                vertex_colors=np.full(vertices.shape, color, dtype=np.uint8),
                vertex_normals=compute_vertex_normals(vertices, indices),
                triangle_indices=indices,
            ),
        )

        return

    # Generate vertices and indices
    vertices = []
    indices = []

    # Generate vertices of the cylinder
    for i in range(resolution):
        theta = 2 * np.pi * i / resolution
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)

        # Generate vertices of the top circle
        top_vertex = np.array([x, height / 2, z])
        vertices.append(top_vertex)

        # Generate vertices of the bottom circle
        bottom_vertex = np.array([x, -height / 2, z])
        vertices.append(bottom_vertex)

        # Generate indices of the sides
        next_i = (i + 1) % resolution
        indices.extend(
            [[2 * i, 2 * next_i, 2 * i + 1], [2 * next_i, 2 * next_i + 1, 2 * i + 1]]
        )

    # Generate vertices of the top and bottom circles
    for y_sign in [1, -1]:
        y = y_sign * height / 2
        center_vertex = np.array([0, y, 0])
        vertices.append(center_vertex)
        center_idx = len(vertices) - 1

        for i in range(resolution):
            next_i = (i + 1) % resolution
            if y_sign == 1:  # Top circle
                indices.append([center_idx, 2 * i, 2 * next_i])
            else:  # Bottom circle
                indices.append([center_idx, 2 * next_i + 1, 2 * i + 1])

    vertices = np.array(vertices)
    indices = np.array(indices)

    global_cylinder_resolution = resolution
    global_cylinder_vertices = vertices
    global_cylinder_indices = indices

    R_object = R.from_quat(orientation)
    vertices = R_object.apply(vertices)
    vertices = vertices + translation

    # Render the cylinder
    rr.log(
        "object/cylinder",
        rr.Mesh3D(
            vertex_positions=vertices,
            vertex_colors=np.full(vertices.shape, color, dtype=np.uint8),
            vertex_normals=compute_vertex_normals(vertices, indices),
            triangle_indices=indices,
        ),
    )


global_ring_resolution = -1
global_ring_vertices = []
global_ring_indices = []


def visualize_ring(
    radius_out=0.05,
    radius_in=0.04,
    height=0.16,
    resolution=128,
    translation=np.zeros(3),
    orientation=np.eye(3),
    color=[0, 255, 0],
):
    global global_ring_resolution
    global global_ring_vertices
    global global_ring_indices

    if global_ring_resolution == resolution:
        vertices = global_ring_vertices
        indices = global_ring_indices

        R_object = R.from_quat(orientation)
        vertices = R_object.apply(vertices)
        vertices = vertices + translation

        # Render the ring
        rr.log(
            "object/ring",
            rr.Mesh3D(
                vertex_positions=vertices,
                vertex_colors=np.full(vertices.shape, color, dtype=np.uint8),
                vertex_normals=compute_vertex_normals(vertices, indices),
                triangle_indices=indices,
            ),
        )

        return

    # Generate vertices of the cylinder
    vertices = []
    indices = []

    # Generate vertices of the sides
    for i in range(resolution):
        theta = 2 * np.pi * i / resolution
        x_out = radius_out * np.cos(theta)
        z_out = radius_out * np.sin(theta)

        x_in = radius_in * np.cos(theta)
        z_in = radius_in * np.sin(theta)

        top_vertex_out = np.array([x_out, height / 2, z_out])
        top_vertex_in = np.array([x_in, height / 2, z_in])

        bottom_vertex_out = np.array([x_out, -height / 2, z_out])
        bottom_vertex_in = np.array([x_in, -height / 2, z_in])

        vertices.append(top_vertex_out)
        vertices.append(top_vertex_in)
        vertices.append(bottom_vertex_out)
        vertices.append(bottom_vertex_in)

        # Generate indices of the sides
        next_i = (i + 1) % resolution
        indices.extend(
            [
                [4 * i, 4 * next_i, 4 * i + 2],
                [4 * next_i, 4 * next_i + 2, 4 * i + 2],
                [4 * i + 1, 4 * next_i + 1, 4 * i + 3],
                [4 * next_i + 1, 4 * next_i + 3, 4 * i + 3],
                [4 * i, 4 * next_i, 4 * i + 1],
                [4 * next_i, 4 * i + 1, 4 * next_i + 1],
                [4 * i + 2, 4 * next_i + 2, 4 * i + 3],
                [4 * next_i + 2, 4 * i + 3, 4 * next_i + 3],
            ]
        )

    vertices = np.array(vertices)
    indices = np.array(indices)

    global_ring_resolution = resolution
    global_ring_vertices = vertices
    global_ring_indices = indices

    R_object = R.from_quat(orientation)
    vertices = R_object.apply(vertices)
    vertices = vertices + translation

    # Render the cylinder
    rr.log(
        "object/ring",
        rr.Mesh3D(
            vertex_positions=vertices,
            vertex_colors=np.full(vertices.shape, color, dtype=np.uint8),
            vertex_normals=compute_vertex_normals(vertices, indices),
            triangle_indices=indices,
        ),
    )


global_prism_resolution = -1
global_prism_vertices = []
global_prism_indices = []


def visualize_prism(
    base_length=0.05,
    height=0.16,
    resolution=128,
    translation=np.zeros(3),
    orientation=np.eye(3),
    color=[0, 255, 0],
):
    global global_prism_resolution
    global global_prism_vertices
    global global_prism_indices

    if global_prism_resolution == resolution:
        vertices = global_prism_vertices
        indices = global_prism_indices

        R_object = R.from_quat(orientation)
        vertices = R_object.apply(vertices)
        vertices = vertices + translation

        # Render the prism
        rr.log(
            "object/prism",
            rr.Mesh3D(
                vertex_positions=vertices,
                vertex_colors=np.full(vertices.shape, color, dtype=np.uint8),
                vertex_normals=compute_vertex_normals(vertices, indices),
                triangle_indices=indices,
            ),
        )

        return

    radius = base_length / (3.0**0.5)
    resolution = 3
    # Generate vertices of the prism
    vertices = []
    indices = []

    # Generate vertices of the sides
    for i in range(resolution):
        theta = 2 * np.pi * i / resolution
        x = radius * np.cos(theta + np.pi / 6.0 + np.pi)
        z = radius * np.sin(theta + np.pi / 6.0 + np.pi)

        # Generate vertices of the top circle
        top_vertex = np.array([x, height / 2, z])
        vertices.append(top_vertex)

        # Generate vertices of the bottom circle
        bottom_vertex = np.array([x, -height / 2, z])
        vertices.append(bottom_vertex)

        # Generate indices of the sides
        next_i = (i + 1) % resolution
        indices.extend(
            [[2 * i, 2 * next_i, 2 * i + 1], [2 * next_i, 2 * next_i + 1, 2 * i + 1]]
        )

    # Generate vertices of the top and bottom circles
    for y_sign in [1, -1]:
        y = y_sign * height / 2
        center_vertex = np.array([0, y, 0])
        vertices.append(center_vertex)
        center_idx = len(vertices) - 1

        for i in range(resolution):
            next_i = (i + 1) % resolution
            if y_sign == 1:  # Top circle
                indices.append([center_idx, 2 * i, 2 * next_i])
            else:  # Bottom circle
                indices.append([center_idx, 2 * next_i + 1, 2 * i + 1])

    vertices = np.array(vertices)
    indices = np.array(indices)

    global_prism_resolution = resolution
    global_prism_vertices = vertices
    global_prism_indices = indices

    R_object = R.from_quat(orientation)
    vertices = R_object.apply(vertices)
    vertices = vertices + translation

    # Render the prism
    rr.log(
        "object/prism",
        rr.Mesh3D(
            vertex_positions=vertices,
            vertex_colors=np.full(vertices.shape, color, dtype=np.uint8),
            vertex_normals=compute_vertex_normals(vertices, indices),
            triangle_indices=indices,
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MANO hand model and object data from a single NPZ file, potentially in multiple Rerun sessions based on metadata."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the .npz file containing hand, object, and metadata.",
    )
    parser.add_argument(
        "--use_hand_local_coordinates",
        type=bool,
        default=False,
        help="Whether to use hand local coordinates.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time in seconds (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="End time in seconds (exclusive). If omitted, visualizes until the end.",
    )
    args = parser.parse_args()

    if args.start < 0:
        parser.error("--start must be >= 0 seconds.")
    if args.end is not None and args.end < 0:
        parser.error("--end must be >= 0 seconds (or omit it).")
    if args.end is not None and args.end < args.start:
        parser.error("--end must be >= --start.")

    data = np.load(args.data_path, allow_pickle=True)
    metadata = data["metadata"].item()

    # Print metadata
    for key, value in metadata.items():
        print(f"data['metadata']['{key}']: {value}")

    # Read metadata
    fps = metadata["fps"]
    mocap_session_name_base = metadata["mocap_session_name"]
    object_size = metadata["object_size"]
    invalid_point_value = metadata["invalid_point_value"]

    is_rubiks_cube = "RubiksCube" in mocap_session_name_base

    # Read data
    optimized_translations = data["hand_translations"]
    optimized_orientations = data["hand_orientations_axis_angle"]
    optimized_poses = data["hand_poses"]
    optimized_shapes = data["hand_shapes"]
    num_frames_hand = optimized_translations.shape[0]

    object_translations = data["object_translations"]
    object_orientations_quat_xyzw = data["object_orientations_quat_xyzw"]
    if is_rubiks_cube:
        object_face_designators = data["object_face_designators"]
        object_rotation_angles = data["object_rotation_angles"]
    num_frames_object = object_translations.shape[0]

    assert num_frames_hand == num_frames_object, (
        f"Number of frames in hand data ({num_frames_hand}) does not match number of frames in object data ({num_frames_object})."
    )

    start_frame = int(math.ceil(args.start * fps))
    end_frame_exclusive = (
        int(math.ceil(args.end * fps)) if args.end is not None else num_frames_hand
    )
    start_frame = max(0, min(start_frame, num_frames_hand))
    end_frame_exclusive = max(start_frame, min(end_frame_exclusive, num_frames_hand))
    if start_frame >= end_frame_exclusive:
        print(
            f"Nothing to visualize: requested [{args.start}, {args.end}) seconds maps to "
            f"empty frame range [{start_frame}, {end_frame_exclusive})."
        )
        return

    print(
        f"\nVisualizing time range: [{args.start}, {args.end}) seconds "
        f"=> frames [{start_frame}, {end_frame_exclusive}) at {fps} FPS."
    )

    # Print data
    for key, value in data.items():
        if key == "metadata":
            continue
        print(f"data['{key}'].shape: {value.shape}")

    # Initialize MANO model and calculate hand vertices and joints
    smplx_model = smplx.create(
        model_path=MANO_MODEL_PATH,
        model_type="mano",
        flat_hand_mean=True,
        is_rhand=False,
        use_pca=False,
        batch_size=num_frames_hand,
    )
    hand_parms = {
        "global_orient": torch.zeros((num_frames_hand, 3), dtype=torch.float32),
        "transl": torch.zeros((num_frames_hand, 3), dtype=torch.float32),
        "hand_pose": torch.tensor(
            optimized_poses.reshape(num_frames_hand, 45), dtype=torch.float32
        ),
        "betas": torch.tensor(
            optimized_shapes.reshape(num_frames_hand, 10), dtype=torch.float32
        ),
    }
    smplx_output = smplx_model(**hand_parms)
    hand_vertices = smplx_output.vertices.detach().cpu().numpy()
    hand_joints = smplx_output.joints.detach().cpu().numpy()
    new_face_index = [
        121,
        214,
        215,
        279,
        239,
        234,
        92,
        38,
        122,
        118,
        117,
        119,
        120,
        108,
        79,
        78,
    ]
    more_face = []  # To make MANO as a whole watertight mesh
    for i in range(2, len(new_face_index)):
        more_face.append([121, new_face_index[i - 1], new_face_index[i]])
    hand_faces = np.concatenate([smplx_model.faces, more_face], axis=0)

    # Convert to hand local coordinates or hand global coordinates
    if args.use_hand_local_coordinates:
        R_hand = R.from_rotvec(optimized_orientations)
        R_hand_inv = R_hand.inv()
        R_object = R.from_quat(object_orientations_quat_xyzw)
        R_object_inv_hand = R_hand_inv * R_object
        object_orientations_quat_xyzw = R_object_inv_hand.as_quat()

        object_translations -= optimized_translations
        object_translations = R_hand_inv.apply(object_translations)
    else:
        for frame_idx in range(num_frames_hand):
            R_hand = R.from_rotvec(optimized_orientations[frame_idx])
            hand_vertices[frame_idx] = (
                R_hand.apply(hand_vertices[frame_idx])
                + optimized_translations[frame_idx]
            )
            hand_joints[frame_idx] = (
                R_hand.apply(hand_joints[frame_idx]) + optimized_translations[frame_idx]
            )

    # Segment the data based on the index_init_frame
    index_init_frame = metadata.get("index_init_frame")
    index_init_frame_segments = []
    if index_init_frame is None:
        index_init_frame_segments = [[0, num_frames_hand]]
    else:
        for segment_idx, segment_bounds in enumerate(index_init_frame):
            if segment_idx == len(index_init_frame) - 1:
                index_init_frame_segments.append([segment_bounds, num_frames_hand])
            else:
                index_init_frame_segments.append(
                    [segment_bounds, index_init_frame[segment_idx + 1]]
                )

    # Visualize each segment
    for segment_idx, segment_bounds in enumerate(index_init_frame_segments):
        segment_data_start_idx = segment_bounds[0]
        segment_data_end_idx = segment_bounds[1]

        segment_viz_start_idx = max(segment_data_start_idx, start_frame)
        segment_viz_end_idx = min(segment_data_end_idx, end_frame_exclusive)
        if segment_viz_start_idx >= segment_viz_end_idx:
            continue

        # Initialize Rerun for this segment
        segment_rerun_name_suffix = (
            f"Segment_{segment_idx}_{segment_viz_start_idx}-{segment_viz_end_idx}"
        )

        init_rerun(
            f"DataViz-{mocap_session_name_base}-{segment_rerun_name_suffix}",
            save=False,
        )
        print(
            f"\nVisualizing Segment {segment_idx}: frames [{segment_viz_start_idx}, {segment_viz_end_idx}) (global indices). "
            f"Rerun session: DataViz-{mocap_session_name_base}-{segment_rerun_name_suffix}"
        )

        colors = [
            [230, 97, 92],
            [88, 196, 157],
            [106, 137, 204],
            [255, 193, 84],
            [186, 123, 202],
            [95, 195, 228],
            [207, 106, 135],
            [139, 195, 74],
            [79, 134, 153],
            [255, 167, 38],
            [149, 117, 205],
            [38, 198, 218],
            [158, 158, 158],
            [121, 85, 72],
            [183, 28, 28],
            [56, 142, 60],
            [63, 81, 181],
        ]

        object_color = colors[np.random.randint(0, len(colors))]
        # Visualize hand for this segment (subsequent frames as partial updates)
        for frame_idx in tqdm(
            range(segment_viz_start_idx, segment_viz_end_idx),
            desc=f"Hand Segment {segment_idx} Updates",
        ):
            if frame_idx >= num_frames_hand:
                break

            rr.set_time("stable_time", duration=frame_idx / fps)
            current_hand_verts = hand_vertices[frame_idx]
            current_hand_normals = compute_vertex_normals(
                current_hand_verts, hand_faces
            )
            rr.log(
                "hand/mesh",
                rr.Mesh3D(
                    vertex_positions=current_hand_verts,
                    triangle_indices=hand_faces,
                    vertex_normals=current_hand_normals,
                    vertex_colors=np.full(
                        current_hand_verts.shape, [255, 255, 255], dtype=np.uint8
                    ),
                ),
            )
            # for i in range(hand_joints.shape[1]):
            #     rr.log(
            #         f"hand/joints/joint_{i}",
            #         rr.Points3D(
            #             positions=hand_joints[frame_idx, i],
            #             colors=np.full(
            #                 hand_joints[frame_idx, i].shape, [0, 255, 0], dtype=np.uint8
            #             ),
            #             radii=0.001,
            #             labels=[f"{i}"],
            #         ),
            #     )

        # Visualize object for this segment
        for frame_idx in tqdm(
            range(segment_viz_start_idx, segment_viz_end_idx),
            desc=f"Object Segment {segment_idx}",
        ):
            if frame_idx >= num_frames_object or is_rubiks_cube:
                break

            if np.allclose(
                object_translations[frame_idx], np.array(invalid_point_value)
            ):
                continue

            rr.set_time(
                "stable_time", duration=frame_idx / fps
            )  # Use global frame_idx for time

            if "Cuboid" in mocap_session_name_base:
                rr.log(
                    "object/cube",
                    rr.Boxes3D(
                        sizes=object_size,
                        centers=object_translations[frame_idx],
                        quaternions=object_orientations_quat_xyzw[frame_idx],
                        colors=object_color,
                        # fill_mode="DenseWireframe",
                        fill_mode="Solid",
                    ),
                )
            elif "Cylinder" in mocap_session_name_base:
                visualize_cylinder(
                    radius=object_size[0] / 2,
                    height=object_size[1],
                    translation=object_translations[frame_idx],
                    orientation=object_orientations_quat_xyzw[frame_idx],
                    color=object_color,
                )
            elif "Plate" in mocap_session_name_base:
                visualize_cylinder(
                    radius=object_size[0] / 2,
                    height=object_size[1],
                    translation=object_translations[frame_idx],
                    orientation=object_orientations_quat_xyzw[frame_idx],
                    color=object_color,
                )
            elif "Ring" in mocap_session_name_base:
                visualize_ring(
                    radius_out=object_size[0] / 2,
                    radius_in=object_size[1] / 2,
                    height=object_size[2],
                    translation=object_translations[frame_idx],
                    orientation=object_orientations_quat_xyzw[frame_idx],
                    color=object_color,
                )
            elif "Prism" in mocap_session_name_base:
                visualize_prism(
                    base_length=object_size[0],
                    height=object_size[1],
                    translation=object_translations[frame_idx],
                    orientation=object_orientations_quat_xyzw[frame_idx],
                    color=object_color,
                )
            else:
                raise ValueError(f"Unknown object type: {mocap_session_name_base}")

        if is_rubiks_cube:
            seg = slice(segment_viz_start_idx, segment_viz_end_idx)
            visualize_rubiks_cube_animation(
                animation_data={
                    "translations": object_translations[seg],
                    "orientations": R.from_quat(
                        object_orientations_quat_xyzw[seg]
                    ).as_matrix(),
                    "face_designators": object_face_designators[seg],
                    "rotation_angles": object_rotation_angles[seg],
                },
                animation_fps=fps,
                cubelet_size=object_size[0] / 2,
                gap=0.0,
                cube_name=f"{mocap_session_name_base}_object",
                time_offset_sec=segment_viz_start_idx / fps,
            )


if __name__ == "__main__":
    main()
