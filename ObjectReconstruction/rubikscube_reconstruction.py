import copy
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import rerun as rr  # For visualization if directly used, or indirectly via rubikscube.py
import transforms3d as t3d
from scipy.spatial.transform import Rotation as R

from HandReconstruction.Utility.utils_visualize import init_rerun
from ObjectReconstruction.rubikscube import (
    RubiksCube222,
    parse_move_string,
    visualize_rubiks_cube_animation,
)

# --- Real data ---
use_real_data = True
save_reconstructed_data = False

num_frames_start = 3870  # rotate start
num_frames_start = 7274  # RUR'U' start
# num_frames_start = 8400  # RUR'U' end
# num_frames_start = 0

num_frames_end = 7500  # RUR'U' end
# num_frames_end = 8400  # RUR'U' end
# num_frames_end = 9000  # RUR'U' end and some random rotation
# num_frames_end = 14965  # all end

_DEFAULT_MOCAP_DIR = os.path.join(
    os.path.dirname(__file__), "MocapData", "RubiksCube_00"
)
mocap_points_path = os.path.join(
    _DEFAULT_MOCAP_DIR, "pts_obj-2025_05_21.npy"
)  # [14966, 384, 3]
facelet_mask_path = os.path.join(
    _DEFAULT_MOCAP_DIR, "obj_face_idx-2025_05_21.npy"
)  # 0-23
patch_mask_path = os.path.join(
    _DEFAULT_MOCAP_DIR, "obj_point_idx-2025_05_21.npy"
)  # 0-15

# --- Parameters for visualization ---
VIZ_LABEL = False
debug_mode = False  # Would log half cube to rerun, and print coplanar scores

# --- Parameters for synthetic data generation ---
NOISE_STD = 0.0  # Reduced from 0.0005
OCCLUSION_PROB = 0.0

# --- Constants for Reconstruction ---
OCCLUSION_PLACEHOLDER = np.array([-1000.0, -1000.0, -1000.0], dtype=np.float32)
NUM_MARKERS_PER_FACELET = 4  # As specified: 16 markers per small square facelet
CUBELET_SIZE = (
    0.0255  # Rubik's Cube size, in meters, consistent with rubikscube.py main
)
GAP = 0.0  # Gap of the cubelets, in meters
FPS = 60
MARKER_SPACING_METERS = (
    0.005  # Spacing between adjacent markers on a facelet, in meters
)
INDEX_COPLANAR_ON_FACE = {
    "U": np.array([7, 10, 22, 19]),
    "D": np.array([13, 16, 4, 1]),
    "L": np.array([6, 18, 12, 0]),
    "R": np.array([21, 9, 3, 15]),
    "F": np.array([20, 23, 17, 14]),
    "B": np.array([11, 8, 2, 5]),
}
# For each face, look at the face directly, the order is like this:
# P.S. you can run this script to visualize the order.
# U Face
# ------
# |7  10|
# |19 22|
# ------
# D Face
# ------
# |13 16|
# |1  4 |
# ------
seeing_horizontal = 100  # Determine starting twisting before seeing_horizontal frames
coplanar_score_threshold = (
    0.01  # If the coplanar score is less than this, then the face is coplanar
)
uncoplanar_score_threshold = (
    0.009  # If the coplanar score is greater than this, then the face is unco-planar
)
angle_snap_to_90_threshold_rad = (
    3 * np.pi / 180
)  # 2 degrees, in radians, to snap to 90 degrees

# --- Helper Functions for Marker Definition ---


def get_local_marker_positions_for_facelet(
    num_markers_side: int = 3, marker_spacing: float = 0.005
) -> np.ndarray:
    """
    Generates local marker positions on a facelet, centered at the origin of the facelet's coordinate system.
    Markers form a grid of num_markers_side x num_markers_side.
    The spacing between adjacent markers in the grid is specified by marker_spacing.

    Parameters:
    - num_markers_side (int): Number of markers along one side of the facelet.
                               Default is 3, for a 3x3 grid.
    - marker_spacing (float): The distance between adjacent markers in meters.
                              Default is 0.005m (5mm).

    Returns:
    - np.ndarray (shape=(num_markers_side*num_markers_side, 3)): Local (x,y,0) coordinates of markers in meters.
                                                                 Returns empty array if num_markers_side < 1.
                                                                 Returns a single marker at [0,0,0] if num_markers_side == 1.
    """
    if num_markers_side < 1:
        return np.empty((0, 3), dtype=np.float32)

    if num_markers_side == 1:  # Centered single marker
        return np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    # Calculate coordinates for one dimension, ensuring the grid is centered
    # Total extent of the marker grid along one dimension
    grid_extent = (num_markers_side - 1) * marker_spacing
    half_extent = grid_extent / 2.0
    lincoords = np.linspace(-half_extent, half_extent, num_markers_side)

    grid_x, grid_y = np.meshgrid(lincoords, lincoords)
    markers = np.vstack(
        [grid_x.ravel(), grid_y.ravel(), np.zeros(num_markers_side**2)]
    ).T
    return markers.astype(np.float32)


def get_reference_cube_marker_data(
    cubelet_size: float = CUBELET_SIZE,
    gap: float = GAP,
    num_markers_per_side_facelet: int = NUM_MARKERS_PER_FACELET,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[Tuple[int, str]],
]:
    """
    Generates the 3D positions and metadata of markers on a canonical, solved Rubik's Cube.
    Markers are placed on each of the 24 external facelets.

    Parameters:
    - cubelet_size (float): The side length of a single small cubelet.
    - gap (float): The visual gap between adjacent cubelets.
    - num_markers_per_side_facelet (int): Number of markers along one side of each small facelet (e.g., 3 for a 3x3 grid).

    Returns:
    - Tuple containing:
        - all_marker_positions_np (np.ndarray, shape=(N_total_markers, 3)):
            3D positions of all markers in the solved cube's frame.
        - all_marker_canonical_face_indices_np (np.ndarray, shape=(N_total_markers,)):
            Canonical facelet index (0-23) for each marker.
        - ordered_external_facelet_identifiers (List[Tuple[int, str]]):
            List of (cubelet_id_numeric, face_key_in_cubelet) for the 24 external facelets, in order
            e.g.
            [(0, 'NX'), (0, 'NY'), (0, 'NZ'), (1, 'PX'), (1, 'NY'), (1, 'NZ'), (2, 'NX'), (2, 'PY'), (2, 'NZ'), (3, 'PX'), (3, 'PY'), (3, 'NZ'), (4, 'NX'), (4, 'NY'), (4, 'PZ'), (5, 'PX'), (5, 'NY'), (5, 'PZ'), (6, 'NX'), (6, 'PY'), (6, 'PZ'), (7, 'PX'), (7, 'PY'), (7, 'PZ')].
    """
    solved_cube = RubiksCube222(cubelet_size=cubelet_size, gap=gap)
    local_facelet_markers = get_local_marker_positions_for_facelet(
        num_markers_per_side_facelet, MARKER_SPACING_METERS
    )

    all_marker_positions_list = []
    all_marker_canonical_face_indices_list = []
    all_marker_cubelet_ids_numeric_list = []
    all_marker_local_positions_in_cubelet_frame_list = []
    ordered_external_facelet_identifiers_list = []
    # For a 2x2x2 cube, there are 8 cubelets, each can have up to 3 external faces. So max 24 facelets.
    MAX_EXPECTED_FACELETS = 24
    canonical_facelet_idx_to_macro_face_designator_map = [None] * MAX_EXPECTED_FACELETS

    total_marker_idx = 0
    current_canonical_facelet_idx_counter = 0

    solved_axis_to_face_char = {
        (0, 1): "R",
        (0, -1): "L",
        (1, 1): "U",
        (1, -1): "D",
        (2, 1): "F",
        (2, -1): "B",
    }

    for cubelet in solved_cube.cubelets:
        cubelet_id_numeric = cubelet["id_numeric"]
        cubelet_id_tuple = cubelet["id_tuple"]

        T_cubelet_in_rubiks = np.eye(4)
        T_cubelet_in_rubiks[:3, :3] = cubelet["current_orientation_in_rubiks"]
        T_cubelet_in_rubiks[:3, 3] = cubelet["current_position_in_rubiks"]

        for face_data in cubelet["faces"]:
            current_face_key = face_data["entity_id_suffix"].split("_")[1]  # e.g., "PX"

            normal_in_cubelet_frame_map = {
                "PX": (0, 1),
                "NX": (0, -1),
                "PY": (1, 1),
                "NY": (1, -1),
                "PZ": (2, 1),
                "NZ": (2, -1),
            }
            normal_in_cubelet = normal_in_cubelet_frame_map[current_face_key]

            is_external = False
            macro_face_designator = None

            world_axis_of_normal = normal_in_cubelet[0]
            world_direction_of_normal = normal_in_cubelet[1]

            if cubelet_id_tuple[world_axis_of_normal] == world_direction_of_normal:
                is_external = True
                macro_face_key = (world_axis_of_normal, world_direction_of_normal)
                macro_face_designator = solved_axis_to_face_char.get(macro_face_key)

            if is_external and macro_face_designator:
                T_face_in_cubelet = face_data["transform_in_cubelet"]

                markers_on_face_local_homog = np.hstack(
                    (
                        local_facelet_markers,
                        np.ones((local_facelet_markers.shape[0], 1)),
                    )
                )

                markers_in_cubelet_homog = (
                    T_face_in_cubelet @ markers_on_face_local_homog.T
                ).T
                markers_in_rubiks_homog = (
                    T_cubelet_in_rubiks @ markers_in_cubelet_homog.T
                ).T

                current_facelet_marker_positions = markers_in_rubiks_homog[:, :3]
                all_marker_positions_list.append(current_facelet_marker_positions)

                ordered_external_facelet_identifiers_list.append(
                    (cubelet_id_numeric, current_face_key)
                )
                # Store the macro face designator for the current canonical facelet index
                if current_canonical_facelet_idx_counter < MAX_EXPECTED_FACELETS:
                    canonical_facelet_idx_to_macro_face_designator_map[
                        current_canonical_facelet_idx_counter
                    ] = macro_face_designator
                else:
                    print(
                        f"Warning: current_canonical_facelet_idx_counter ({current_canonical_facelet_idx_counter}) exceeds MAX_EXPECTED_FACELETS ({MAX_EXPECTED_FACELETS})."
                    )

                num_markers_on_this_facelet = current_facelet_marker_positions.shape[0]
                for i_marker_on_facelet in range(num_markers_on_this_facelet):
                    all_marker_canonical_face_indices_list.append(
                        current_canonical_facelet_idx_counter
                    )
                    all_marker_cubelet_ids_numeric_list.append(cubelet_id_numeric)
                    all_marker_local_positions_in_cubelet_frame_list.append(
                        markers_in_cubelet_homog[i_marker_on_facelet, :3].copy()
                    )
                    total_marker_idx += 1

                current_canonical_facelet_idx_counter += 1

    if not all_marker_positions_list:
        print("Warning: No external markers found for the reference cube.")
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=int),
            [],
        )

    all_marker_positions_np = np.vstack(all_marker_positions_list).astype(np.float32)
    all_marker_canonical_face_indices_np = np.array(
        all_marker_canonical_face_indices_list, dtype=int
    )

    return (
        all_marker_positions_np,
        all_marker_canonical_face_indices_np,
        ordered_external_facelet_identifiers_list,
    )


def log_mocap_points(
    sim_marker_positions_world: np.ndarray,
    sim_marker_visibility: np.ndarray,
    model_marker_to_canonical_face_index_map: np.ndarray,
    total_frames: int,
    fps: int,
    viz_label: bool = True,
):
    """
    Log mocap marker points to Rerun, coloring by canonical face index and clearing invisible markers.

    Args:
        sim_marker_positions_world (np.ndarray, shape=(N_frames, N_markers, 3)): The 3D positions of all observed markers over N_frames.
        sim_marker_visibility (np.ndarray, shape=(N_frames, N_markers)): The visibility of all markers over N_frames (bool).
        model_marker_to_canonical_face_index_map (np.ndarray, shape=(N_markers,)): The mapping of markers to faces.
        total_frames (int): Total number of frames.
        fps (int): Frames per second.
        viz_label (bool): Whether to visualize labels for each marker.
    """
    # Get the canonical face index for each marker
    marker_to_face_idx_map = np.array(model_marker_to_canonical_face_index_map)
    unique_face_indices = np.unique(marker_to_face_idx_map)
    num_unique_faces = len(unique_face_indices)  # 24

    # Create a color map for unique face indices
    colormap = {}

    if num_unique_faces > 0:
        # Create a mapping from facelet index to face name
        facelet_to_face_map = {}
        for face_name, facelet_indices in INDEX_COPLANAR_ON_FACE.items():
            for facelet_idx in facelet_indices:
                facelet_to_face_map[facelet_idx] = face_name

        # Define specific colors for each face (RGB values 0-255)
        face_colors = {
            "U": np.array([255, 255, 0], dtype=np.uint8),  # Yellow
            "D": np.array([255, 255, 255], dtype=np.uint8),  # White
            "L": np.array([255, 165, 0], dtype=np.uint8),  # Orange
            "R": np.array([255, 0, 0], dtype=np.uint8),  # Red
            "F": np.array([0, 0, 255], dtype=np.uint8),  # Blue
            "B": np.array([0, 255, 0], dtype=np.uint8),  # Green
        }

        # Assign colors to each unique face index
        for face_idx in unique_face_indices:
            if face_idx in facelet_to_face_map:
                face_name = facelet_to_face_map[face_idx]
                colormap[face_idx] = face_colors.get(
                    face_name, np.array([128, 128, 128], dtype=np.uint8)
                )  # Default gray
            else:
                # Fallback for unmapped indices
                colormap[face_idx] = np.array([128, 128, 128], dtype=np.uint8)  # Gray

    for i_frame in range(total_frames):
        current_time_sec = i_frame / float(fps)
        rr.set_time("stable_time", duration=current_time_sec)

        visible_markers_mask = sim_marker_visibility[i_frame]  # (N_total_markers,)

        for face_idx in unique_face_indices:
            markers_belonging_to_face_mask = marker_to_face_idx_map == face_idx
            markers_belonging_to_face_and_visible_mask = (
                markers_belonging_to_face_mask & visible_markers_mask
            )
            if not viz_label:
                point = sim_marker_positions_world[
                    i_frame, markers_belonging_to_face_and_visible_mask
                ]
                point_num = point.shape[0]

                if point_num == 0:
                    rr.log(
                        f"mocap_points/{face_idx}",
                        rr.Clear(recursive=True),
                    )
                else:
                    rr.log(
                        f"mocap_points/{face_idx}",
                        rr.Points3D(
                            positions=point,
                            colors=np.tile(colormap[face_idx], (point_num, 1)),
                            radii=0.0005,
                        ),
                    )
            else:
                markers_belonging_to_face_and_visible_indices = np.where(
                    markers_belonging_to_face_and_visible_mask
                )[0]
                markers_belonging_to_face_and_not_visible_indices = np.where(
                    markers_belonging_to_face_mask & ~visible_markers_mask
                )[0]

                for exist_point_idx in markers_belonging_to_face_and_visible_indices:
                    rr.log(
                        f"mocap_points/{face_idx}/{exist_point_idx % 16}",
                        rr.Points3D(
                            positions=point[exist_point_idx],
                            colors=np.tile(colormap[face_idx], (1, 1)),
                            radii=0.0005,
                            labels=[f"{face_idx}_{exist_point_idx % 16}"],
                        ),
                    )
                for (
                    not_exist_point_idx
                ) in markers_belonging_to_face_and_not_visible_indices:
                    rr.log(
                        f"mocap_points/{face_idx}/{not_exist_point_idx % 16}",
                        rr.Clear(recursive=True),
                    )


def rotate_rubiks_face(
    face_designator: str, index_on_face: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Rotate the Rubik's Cube face by 90 degrees clockwise.

    Args:
        face_designator (str): The face to rotate. Includes [U, D, L, R, F, B, U', D', L', R', F', B']
        index_on_face (Dict[str, np.ndarray]): The index of the markers on the face.
    """
    new_index_on_face = copy.deepcopy(index_on_face)
    if face_designator == "U":
        new_index_on_face["U"] = index_on_face["U"][[3, 0, 1, 2]]
        new_index_on_face["F"][[0, 1]] = index_on_face["R"][[0, 1]]
        new_index_on_face["R"][[0, 1]] = index_on_face["B"][[0, 1]]
        new_index_on_face["B"][[0, 1]] = index_on_face["L"][[0, 1]]
        new_index_on_face["L"][[0, 1]] = index_on_face["F"][[0, 1]]
    elif face_designator == "D":
        new_index_on_face["D"] = index_on_face["D"][[3, 0, 1, 2]]
        new_index_on_face["F"][[2, 3]] = index_on_face["L"][[2, 3]]
        new_index_on_face["L"][[2, 3]] = index_on_face["B"][[2, 3]]
        new_index_on_face["B"][[2, 3]] = index_on_face["R"][[2, 3]]
        new_index_on_face["R"][[2, 3]] = index_on_face["F"][[2, 3]]
    elif face_designator == "L":
        new_index_on_face["L"] = index_on_face["L"][[3, 0, 1, 2]]
        new_index_on_face["U"][[0, 3]] = index_on_face["B"][[2, 1]]
        new_index_on_face["B"][[1, 2]] = index_on_face["D"][[3, 0]]
        new_index_on_face["D"][[0, 3]] = index_on_face["F"][[0, 3]]
        new_index_on_face["F"][[0, 3]] = index_on_face["U"][[0, 3]]
    elif face_designator == "R":
        new_index_on_face["R"] = index_on_face["R"][[3, 0, 1, 2]]
        new_index_on_face["U"][[1, 2]] = index_on_face["F"][[1, 2]]
        new_index_on_face["F"][[1, 2]] = index_on_face["D"][[1, 2]]
        new_index_on_face["D"][[1, 2]] = index_on_face["B"][[3, 0]]
        new_index_on_face["B"][[0, 3]] = index_on_face["U"][[2, 1]]
    elif face_designator == "F":
        new_index_on_face["F"] = index_on_face["F"][[3, 0, 1, 2]]
        new_index_on_face["U"][[2, 3]] = index_on_face["L"][[1, 2]]
        new_index_on_face["L"][[1, 2]] = index_on_face["D"][[0, 1]]
        new_index_on_face["D"][[0, 1]] = index_on_face["R"][[3, 0]]
        new_index_on_face["R"][[0, 3]] = index_on_face["U"][[3, 2]]
    elif face_designator == "B":
        new_index_on_face["B"] = index_on_face["B"][[3, 0, 1, 2]]
        new_index_on_face["U"][[0, 1]] = index_on_face["R"][[1, 2]]
        new_index_on_face["R"][[1, 2]] = index_on_face["D"][[2, 3]]
        new_index_on_face["D"][[2, 3]] = index_on_face["L"][[3, 0]]
        new_index_on_face["L"][[0, 3]] = index_on_face["U"][[1, 0]]
    elif face_designator.endswith("'"):
        new_index_on_face = rotate_rubiks_face(face_designator[0], new_index_on_face)
        new_index_on_face = rotate_rubiks_face(face_designator[0], new_index_on_face)
        new_index_on_face = rotate_rubiks_face(face_designator[0], new_index_on_face)
    else:
        raise ValueError(f"Invalid face designator: {face_designator}")
    return new_index_on_face


# --- Simulation Data Generation ---


def generate_simulated_mocap_data(
    conceptual_moves: List[Dict[str, Any]],
    fps: int = 60,
    cubelet_size: float = CUBELET_SIZE,
    gap: float = GAP,
    num_markers_per_side_facelet: int = NUM_MARKERS_PER_FACELET,
    noise_stddev: float = 0.001,  # meters, if cubelet_size is in meters
    occlusion_rate: float = 0.1,  # probability of a marker being occluded per frame
    global_motion: bool = True,
) -> Dict[str, Any]:
    """
    Generates simulated motion capture data for a Rubik's Cube animation.

    Parameters:
    - conceptual_moves (List[Dict[str, Any]]): List of moves, e.g.,
        `[{"move_str": "U", "duration_sec": 1.0}, {"move_str": "R'", "duration_sec": 0.5}]`
    - fps (int): Frames per second for the animation.
    - cubelet_size (float): Side length of a single cubelet.
    - gap (float): Gap between cubelets.
    - num_markers_per_side_facelet (int): Markers per side of a facelet (e.g., 3 for 3x3 grid).
    - noise_stddev (float): Standard deviation of Gaussian noise added to marker positions.
    - occlusion_rate (float): Probability (0-1) that any given marker is occluded in a frame.
    - global_motion (bool): Whether to apply global motion to the simulation.

    Returns:
    - Dict[str, Any]: A dictionary containing:
        - 'total_frames' (int): Total number of frames in the animation.
        - 'fps' (int): Frames per second.
        - 'reference_marker_canonical_face_idx' (np.ndarray): (N_total_markers,) canonical facelet indices for each marker.
        - 'simulated_marker_positions_world' (np.ndarray): (N_frames, N_total_markers, 3) observed positions.
        - 'simulated_marker_visibility' (np.ndarray): (N_frames, N_total_markers) boolean visibility.
        - 'ground_truth_animation': Dict containing 'translations', 'orientations',
                                     'face_designators', 'rotation_angles' from the simulation.
    """
    # 1. Initialize Rubik's Cube and Reference Markers
    sim_cube = RubiksCube222(cubelet_size=cubelet_size, gap=gap)
    (
        ref_marker_positions_solved_frame,
        ref_marker_canonical_face_idx,
        ordered_external_facelets_identifiers,
    ) = get_reference_cube_marker_data(cubelet_size, gap, num_markers_per_side_facelet)

    num_total_markers = ref_marker_positions_solved_frame.shape[0]

    if num_total_markers == 0:
        raise ValueError(
            "No reference markers generated. Check get_reference_cube_marker_data."
        )

    # 2. Calculate total animation duration and frames (similar to rubikscube.py main)
    if not conceptual_moves:
        total_animation_duration_sec = 0
    else:
        total_animation_duration_sec = sum(
            move.get("duration_sec", 0) for move in conceptual_moves
        )

    total_frames = int(round(total_animation_duration_sec * fps))

    if total_frames == 0 and total_animation_duration_sec > 0:
        print(
            f"Warning: Total duration {total_animation_duration_sec}s and FPS {fps} results in 0 frames. Simulating 1 static frame."
        )
        total_frames = 1  # Ensure at least one frame for static poses
    elif total_frames == 0 and total_animation_duration_sec == 0:
        # print(
        #     "Warning: No conceptual moves and zero duration. Simulating 1 static frame."
        # )
        total_frames = 1

    # 3. Generate ground truth per-frame rotations and global transformations (from rubikscube.py logic)
    gt_face_designators = np.full(total_frames, None, dtype=object)
    gt_rotation_angles = np.zeros(total_frames, dtype=float)

    current_frame_idx = 0
    if total_animation_duration_sec > 0:  # Only process moves if there's duration
        for move_def in conceptual_moves:
            move_str = move_def["move_str"]
            duration_sec = move_def["duration_sec"]

            face_designator, total_angle_rad, _ = parse_move_string(move_str)
            num_frames_for_this_move = int(round(duration_sec * fps))

            if (
                num_frames_for_this_move == 0 and duration_sec > 0
            ):  # Avoid division by zero if duration is too short
                print(
                    f"Warning: Move '{move_str}' duration {duration_sec}s is too short for FPS {fps}, results in 0 frames. Skipping rotation part of this move."
                )
                current_frame_idx += 0  # Effectively, this move segment contributes 0 frames of rotation.
                # The time still passes for global motion.
                # To ensure current_frame_idx advances correctly for global motion based on total_frames:
                # This part of logic needs to align with how global transforms are made.
                # Let's assume global transforms are generated for 'total_frames' irrespective of move specifics.
                # The key is that sum of num_frames_for_this_move should approximate total_frames.

            if (
                face_designator
                and total_angle_rad != 0
                and num_frames_for_this_move > 0
            ):
                angle_per_frame = total_angle_rad / num_frames_for_this_move
                for i in range(num_frames_for_this_move):
                    frame_to_apply = current_frame_idx + i
                    if frame_to_apply < total_frames:
                        gt_face_designators[frame_to_apply] = face_designator
                        gt_rotation_angles[frame_to_apply] = angle_per_frame
            current_frame_idx += num_frames_for_this_move
        # Ensure current_frame_idx doesn't exceed total_frames due to rounding
        # This loop structure might need refinement if sum of num_frames_for_this_move != total_frames

    # Ensure gt arrays match total_frames, handling empty conceptual_moves
    if len(gt_face_designators) != total_frames:
        gt_face_designators = np.resize(gt_face_designators, total_frames)
        if (
            total_frames > 0 and gt_face_designators[0] is None and conceptual_moves
        ):  # Fill if resized due to mismatch
            if gt_face_designators.dtype == object:
                gt_face_designators.fill(None)

    if len(gt_rotation_angles) != total_frames:
        gt_rotation_angles = np.resize(gt_rotation_angles, total_frames)
        if (
            total_frames > 0 and gt_rotation_angles[0] == 0.0 and conceptual_moves
        ):  # Fill if resized
            gt_rotation_angles.fill(0.0)

    # Example global transformations (sine waves, like in rubikscube.py main)
    gt_translations = np.zeros((total_frames, 3))
    gt_orientations_mat = np.zeros((total_frames, 3, 3))
    gt_orientations_mat[:, :, :] = np.eye(3)

    if global_motion:
        for i in range(total_frames):
            t_norm = i / (total_frames - 1) if total_frames > 1 else 0.0

            gt_translations[i, 0] = 0.05 * np.sin(
                1 * np.pi * t_norm
            )  # Slower global motion
            gt_translations[i, 1] = 0.03 * np.cos(2 * np.pi * t_norm)
            # gt_translations[i, 2] = 0.0 # Keep it on a plane for easier debugging initially

            angle_z = 0.3 * np.pi * t_norm
            angle_x = 0.1 * np.pi * np.sin(1 * np.pi * t_norm)
            angle_y = 0.5 * np.pi * np.cos(2 * np.pi * t_norm)

            Rz = t3d.axangles.axangle2mat([0, 0, 1], angle_z)
            Rx = t3d.axangles.axangle2mat([1, 0, 0], angle_x)
            Ry = t3d.axangles.axangle2mat([0, 1, 0], angle_y)
            gt_orientations_mat[i] = Rz @ Rx @ Ry

    ground_truth_animation_dict = {
        "translations": gt_translations,
        "orientations": gt_orientations_mat,
        "face_designators": gt_face_designators,
        "rotation_angles": gt_rotation_angles,
    }

    # 4. Simulate frame by frame, get marker positions
    simulated_marker_positions_world = np.full(
        (total_frames, num_total_markers, 3), OCCLUSION_PLACEHOLDER, dtype=np.float32
    )
    simulated_marker_visibility = np.zeros(
        (total_frames, num_total_markers), dtype=bool
    )

    # Create a temporary Rubik's cube for simulation state
    # The ref_marker_info links marker_id_global to its cubelet and facelet origin.
    # The ref_marker_positions are in the solved cube's frame.
    # For each frame, we need to transform these ref_marker_positions based on:
    #   1. Cubelet's current state (pos/ori within Rubik's)
    #   2. Rubik's global state (pos/ori in world)

    # The simulation loop now iterates through `ordered_external_facelets_identifiers`
    # to ensure markers are generated in the correct order and only for these facelets.
    local_facelet_markers = get_local_marker_positions_for_facelet(
        num_markers_per_side_facelet, MARKER_SPACING_METERS
    )
    local_facelet_markers_homog = np.hstack(
        (
            local_facelet_markers,
            np.ones((local_facelet_markers.shape[0], 1)),
        )
    ).T  # Shape (4, N_markers_per_facelet) for matmul

    for i_frame in range(total_frames):
        # Apply incremental rotation to sim_cube for this frame
        face_des = gt_face_designators[i_frame]
        inc_angle = gt_rotation_angles[i_frame]
        if face_des and inc_angle != 0.0:
            sim_cube.rotate_rubiks_face(face_des, inc_angle)

        # Get global transform for this frame
        T_rubiks_in_world_overall = np.eye(4)
        T_rubiks_in_world_overall[:3, :3] = gt_orientations_mat[i_frame]
        T_rubiks_in_world_overall[:3, 3] = gt_translations[i_frame]

        current_frame_marker_idx = 0
        # Iterate through the defined external facelets in their canonical order
        for ref_cubelet_id, ref_face_key in ordered_external_facelets_identifiers:
            # Find the current state of this specific cubelet (ref_cubelet_id) in the sim_cube
            sim_cubelet_state = None
            for c_state in sim_cube.cubelets:
                if c_state["id_numeric"] == ref_cubelet_id:
                    sim_cubelet_state = c_state
                    break

            if sim_cubelet_state is None:
                print(
                    f"Error: Could not find sim_cubelet_state for ref_cubelet_id {ref_cubelet_id}"
                )
                # This should not happen if cubelet IDs are consistent
                continue

            # Transform of the cubelet in the Rubik's cube's frame
            T_cubelet_in_rubiks = np.eye(4)
            T_cubelet_in_rubiks[:3, :3] = sim_cubelet_state[
                "current_orientation_in_rubiks"
            ]
            T_cubelet_in_rubiks[:3, 3] = sim_cubelet_state["current_position_in_rubiks"]
            T_cubelet_in_world = T_rubiks_in_world_overall @ T_cubelet_in_rubiks

            # Find the current state of this specific face (ref_face_key) in the sim_cubelet_state
            sim_face_data = None
            for f_data in sim_cubelet_state["faces"]:
                if f_data["entity_id_suffix"].split("_")[1] == ref_face_key:
                    sim_face_data = f_data
                    break

            if sim_face_data is None:
                print(
                    f"Error: Could not find sim_face_data for ref_face_key {ref_face_key} in cubelet {ref_cubelet_id}"
                )
                # This should not happen
                continue

            T_face_in_cubelet_sim = sim_face_data["transform_in_cubelet"]
            T_final_face_in_world = T_cubelet_in_world @ T_face_in_cubelet_sim

            transformed_markers_homog = (
                T_final_face_in_world @ local_facelet_markers_homog
            )
            transformed_markers_world = transformed_markers_homog[
                :3, :
            ].T  # (N_markers_facelet, 3)

            num_markers_on_this_facelet = transformed_markers_world.shape[0]
            for i_m_on_f in range(num_markers_on_this_facelet):
                if current_frame_marker_idx < num_total_markers:
                    raw_pos = transformed_markers_world[i_m_on_f]
                    noise = np.random.normal(0, noise_stddev, 3)
                    noisy_pos = raw_pos + noise

                    if np.random.rand() < occlusion_rate:
                        simulated_marker_positions_world[
                            i_frame, current_frame_marker_idx, :
                        ] = OCCLUSION_PLACEHOLDER
                        simulated_marker_visibility[
                            i_frame, current_frame_marker_idx
                        ] = False
                    else:
                        simulated_marker_positions_world[
                            i_frame, current_frame_marker_idx, :
                        ] = noisy_pos
                        simulated_marker_visibility[
                            i_frame, current_frame_marker_idx
                        ] = True
                    current_frame_marker_idx += 1
                else:
                    print(
                        f"Warning: Exceeded num_total_markers ({num_total_markers}) at frame {i_frame}, "
                        f"marker index {current_frame_marker_idx} while processing facelet "
                        f"({ref_cubelet_id}, {ref_face_key})."
                    )
                    break  # Break from markers_on_facelet loop
            if current_frame_marker_idx >= num_total_markers:
                break  # Break from ordered_external_facelets_identifiers loop

        if (
            current_frame_marker_idx < num_total_markers
            and total_frames > 0
            and num_total_markers > 0
        ):
            # This warning might indicate that not all ordered_external_facelets_identifiers were processed
            # or some issue in the loop logic if num_total_markers is not reached.
            print(
                f"Warning: Frame {i_frame}: Processed {current_frame_marker_idx} markers, expected {num_total_markers}. "
                f"This might occur if ordered_external_facelets_identifiers is shorter than expected or cube state is unusual."
            )

    return {
        "total_frames": total_frames,
        "fps": fps,
        "reference_marker_canonical_face_idx": ref_marker_canonical_face_idx,
        "simulated_marker_positions_world": simulated_marker_positions_world,
        "simulated_marker_visibility": simulated_marker_visibility,
        "ground_truth_animation_data": ground_truth_animation_dict,
    }


# --- Reconstruction Algorithm ---


def calculate_coplanar_score(
    marker_positions: np.ndarray,
    marker_visibility: np.ndarray,
    marker_to_face_index_map: np.ndarray,
    index_coplanar_on_face: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Calculate the coplanar score of the marker positions for major Rubik's Cube faces.
    A lower score indicates better coplanarity (closer to zero is best).

    Args:
        marker_positions (np.ndarray, shape=(N_frames, N_markers, 3)): The 3D positions of all markers over N_frames.
        marker_visibility (np.ndarray, shape=(N_frames, N_markers)): Boolean visibility of each marker in each frame.
        marker_to_face_index_map (np.ndarray, shape=(N_markers,)): Maps each global marker index
            to its canonical facelet index (0-23).
        index_coplanar_on_face (Dict[str, np.ndarray], each array with shape: (4,)):
            A dictionary where keys are macro face designators (e.g., "U", "D")
            and values are numpy arrays of 4 canonical facelet indices that constitute that macro face.
            These are the facelets whose collective markers should be coplanar.

    Returns:
        coplanar_score (np.ndarray, shape=(N_frames, 30)): The coplanarity score for each of the 6 big faces and their 01, 12, 23, 30 sub-faces
            The output is a coplanarity score array of shape (N_frames, 30), with the following order for each frame:
            [U, U01, U12, U23, U30, D, D01, D12, D23, D30, L, L01, L12, L23, L30, R, R01, R12, R23, R30, F, F01, F12, F23, F30, B, B01, B12, B23, B30].
            For each macro face (U, D, L, R, F, B) and its four adjacent facelet pairs (e.g., U01, U12, U23, U30), the score is computed.
            The score is defined as the smallest singular value from PCA (SVD) of the visible marker points on the corresponding face or facelet pair.
            If all points are located on a single facelet (i.e., not enough spatial diversity), the score is set to np.inf to indicate the result is not meaningful.
    """
    N_frames = marker_positions.shape[0]
    if N_frames == 0:
        return np.empty((0, 30), dtype=np.float32)

    N_total_markers = marker_positions.shape[1]
    # Define the standard order for the 6 macro faces in the output
    face_keys_ordered = [
        "U",
        "D",
        "L",
        "R",
        "F",
        "B",
    ]
    num_macro = 30

    if N_total_markers == 0:  # No markers to process
        return np.full((N_frames, num_macro), np.inf, dtype=np.float32)

    coplanar_scores_all_frames = np.full(
        (N_frames, num_macro), np.inf, dtype=np.float32
    )

    def calculate_coplanar_score_point_cloud(
        points_array: np.ndarray,
    ):
        num_relevant_points = points_array.shape[0]

        if num_relevant_points < 3:
            return np.inf
        else:
            centroid = np.mean(points_array, axis=0)
            centered_points = points_array - centroid

            try:
                _, s_values, _ = np.linalg.svd(centered_points)
                # s_values has shape (min(num_relevant_points, 3),)
                # Since num_relevant_points >= 3, s_values will have 3 elements if matrix is full rank.
                # s_values[2] is the smallest singular value.
                return s_values[2]
            except np.linalg.LinAlgError:
                return np.inf

    for i_frame in range(N_frames):
        for i_face_key_idx, macro_face_key in enumerate(face_keys_ordered):
            facelet_id = index_coplanar_on_face[macro_face_key]

            facelet_1_mask = np.isin(marker_to_face_index_map, facelet_id[0])
            facelet_1_valid_mask = facelet_1_mask & marker_visibility[i_frame, :]
            facelet_1_points = marker_positions[i_frame, facelet_1_valid_mask, :]
            facelet_1_has_points = facelet_1_points.shape[0] > 0

            facelet_2_mask = np.isin(marker_to_face_index_map, facelet_id[1])
            facelet_2_valid_mask = facelet_2_mask & marker_visibility[i_frame, :]
            facelet_2_points = marker_positions[i_frame, facelet_2_valid_mask, :]
            facelet_2_has_points = facelet_2_points.shape[0] > 0

            facelet_3_mask = np.isin(marker_to_face_index_map, facelet_id[2])
            facelet_3_valid_mask = facelet_3_mask & marker_visibility[i_frame, :]
            facelet_3_points = marker_positions[i_frame, facelet_3_valid_mask, :]
            facelet_3_has_points = facelet_3_points.shape[0] > 0

            facelet_4_mask = np.isin(marker_to_face_index_map, facelet_id[3])
            facelet_4_valid_mask = facelet_4_mask & marker_visibility[i_frame, :]
            facelet_4_points = marker_positions[i_frame, facelet_4_valid_mask, :]
            facelet_4_has_points = facelet_4_points.shape[0] > 0

            if (
                facelet_1_has_points
                + facelet_2_has_points
                + facelet_3_has_points
                + facelet_4_has_points
                > 2
            ):
                coplanar_scores_all_frames[i_frame, i_face_key_idx * 5] = (
                    calculate_coplanar_score_point_cloud(
                        np.concatenate(
                            [
                                facelet_1_points,
                                facelet_2_points,
                                facelet_3_points,
                                facelet_4_points,
                            ],
                            axis=0,
                        )
                    )
                )
            if facelet_1_has_points and facelet_2_has_points:
                coplanar_scores_all_frames[i_frame, i_face_key_idx * 5 + 1] = (
                    calculate_coplanar_score_point_cloud(
                        np.concatenate([facelet_1_points, facelet_2_points], axis=0)
                    )
                )
            if facelet_2_has_points and facelet_3_has_points:
                coplanar_scores_all_frames[i_frame, i_face_key_idx * 5 + 2] = (
                    calculate_coplanar_score_point_cloud(
                        np.concatenate([facelet_2_points, facelet_3_points], axis=0)
                    )
                )
            if facelet_3_has_points and facelet_4_has_points:
                coplanar_scores_all_frames[i_frame, i_face_key_idx * 5 + 3] = (
                    calculate_coplanar_score_point_cloud(
                        np.concatenate([facelet_3_points, facelet_4_points], axis=0)
                    )
                )
            if facelet_4_has_points and facelet_1_has_points:
                coplanar_scores_all_frames[i_frame, i_face_key_idx * 5 + 4] = (
                    calculate_coplanar_score_point_cloud(
                        np.concatenate([facelet_4_points, facelet_1_points], axis=0)
                    )
                )

    return coplanar_scores_all_frames


def get_twisting_designator_from_coplanar_score(
    coplanar_scores: np.ndarray,
    coplanar_score_threshold: float = coplanar_score_threshold,
    uncoplanar_score_threshold: float = uncoplanar_score_threshold,
) -> str:
    """
    Get the twisting designator from the coplanar score.

    Find the first frame that 2 faces are coplanar and the other 4 faces are unco-planar.
    If there is no such frame, return None.

    Args:
        coplanar_scores (np.ndarray, shape=(N_frames, 30)): The coplanarity score for each of the 6 macro faces
            (ordered U, U01, U12, U23, U30, D, D01, D12, D23, D30, L, L01, L12, L23, L30, R, R01, R12, R23, R30, F, F01, F12, F23, F30, B, B01, B12, B23, B30) for each frame. Score is the smallest singular value from PCA of
            the visible points on the face; np.inf if < 3 points are visible or SVD fails.
        coplanar_score_threshold (float): The threshold for coplanar score.
        uncoplanar_score_threshold (float): The threshold for unco-planar score.

    Returns:
        twisting_designator (str): The nearest twisting designator. ["U", "R", "F"]
        nearest_frame_offset_with_rotation (int): The index of the frame that satisfies the condition.
    """
    U_coplanar_indices = np.concatenate(
        [
            np.array([0, 1, 2, 3, 4]),  # U
            np.array([5, 6, 7, 8, 9]),  # D
            np.array([11, 13]),  # L
            np.array([16, 18]),  # R
            np.array([21, 23]),  # F
            np.array([26, 28]),  # B
        ]
    )  # If U is rotating, these indices of `coplanar_scores` should be coplanar
    U_uncoplanar_indices = np.concatenate(
        [
            # np.array([0, 1, 2, 3, 4]),  # U
            # np.array([5, 6, 7, 8, 9]),  # D
            np.array([10, 12, 14]),  # L
            np.array([15, 17, 19]),  # R
            np.array([20, 22, 24]),  # F
            np.array([25, 27, 29]),  # B
        ]
    )  # If U is rotating, these indices of `coplanar_scores` should be unco-planar
    R_coplanar_indices = np.concatenate(
        [
            np.array([2, 4]),  # U
            np.array([7, 9]),  # D
            np.array([10, 11, 12, 13, 14]),  # L
            np.array([15, 16, 17, 18, 19]),  # R
            np.array([22, 24]),  # F
            np.array([27, 29]),  # B
        ]
    )  # If R is rotating, these indices of `coplanar_scores` should be coplanar
    R_uncoplanar_indices = np.concatenate(
        [
            np.array([0, 1, 3]),  # U
            np.array([5, 6, 8]),  # D
            # np.array([10, 11, 12, 13, 14]),  # L
            # np.array([15, 16, 17, 18, 19]),  # R
            np.array([20, 21, 23]),  # F
            np.array([25, 26, 28]),  # B
        ]
    )  # If R is rotating, these indices of `coplanar_scores` should be unco-planar
    F_coplanar_indices = np.concatenate(
        [
            np.array([1, 3]),  # U
            np.array([6, 8]),  # D
            np.array([12, 14]),  # L
            np.array([17, 19]),  # R
            np.array([20, 21, 22, 23, 24]),  # F
            np.array([25, 26, 27, 28, 29]),  # B
        ]
    )  # If F is rotating, these indices of `coplanar_scores` should be coplanar
    F_uncoplanar_indices = np.concatenate(
        [
            np.array([0, 2, 4]),  # U
            np.array([5, 7, 9]),  # D
            np.array([10, 11, 13]),  # L
            np.array([15, 16, 18]),  # R
            # np.array([20, 21, 22, 23, 24]),  # F
            # np.array([25, 26, 27, 28, 29]),  # B
        ]
    )  # If F is rotating, these indices of `coplanar_scores` should be unco-planar
    for i_frame in range(coplanar_scores.shape[0]):
        coplanar_scores_for_this_frame = coplanar_scores[i_frame, :]
        coplanar_face_indices = np.where(
            (coplanar_scores_for_this_frame < coplanar_score_threshold)
            | (coplanar_scores_for_this_frame == np.inf)
        )[0]
        unco_planar_face_indices = np.where(
            coplanar_scores_for_this_frame > uncoplanar_score_threshold
        )[0]
        # print(coplanar_scores_for_this_frame)
        # print(coplanar_face_indices)
        # print(unco_planar_face_indices)
        # exit()

        twisting_designator = []
        # Check if all indices in U_coplanar_indices are present in coplanar_face_indices
        if np.all(np.isin(U_coplanar_indices, coplanar_face_indices)) and np.all(
            np.isin(U_uncoplanar_indices, unco_planar_face_indices)
        ):
            twisting_designator.append("U")

        if np.all(np.isin(R_coplanar_indices, coplanar_face_indices)) and np.all(
            np.isin(R_uncoplanar_indices, unco_planar_face_indices)
        ):
            twisting_designator.append("R")

        if np.all(np.isin(F_coplanar_indices, coplanar_face_indices)) and np.all(
            np.isin(F_uncoplanar_indices, unco_planar_face_indices)
        ):
            twisting_designator.append("F")

        if len(twisting_designator) > 1:
            print("Warning: Multiple twisting designators found")
            return twisting_designator[0], i_frame
        elif len(twisting_designator) == 1:
            return twisting_designator[0], i_frame
        else:
            pass
    return None, None


def separate_122_block_from_marker_positions(
    marker_positions: np.ndarray,
    marker_visibility: np.ndarray,
    marker_to_face_index_map: np.ndarray,
    designator: str,
    index_on_face: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate the 222 block point cloud to 2 blocks, each of which is a 122 block point cloud.

    The first block is always D, L, B facelets.
    The second block is always U, R, F facelets.

    Args:
        marker_positions (np.ndarray, shape=(N_frames, N_markers, 3)): The 3D positions of all markers over N_frames.
        marker_visibility (np.ndarray, shape=(N_frames, N_markers)): Boolean visibility of each marker in each frame.
        marker_to_face_index_map (np.ndarray, shape=(N_markers,)): Maps each global marker index to its canonical facelet index (0-23).
        designator (str): The designator of the twisting block. e.g., "U", "R", "F".
        index_on_face (Dict[str, np.ndarray], each array with shape: (4,)): A dictionary where keys are macro face designators (e.g., "U", "D")
            and values are numpy arrays of 4 canonical facelet indices that constitute that macro face.
            e.g.,
            INDEX_COPLANAR_ON_FACE = {
                "U": np.array([7, 10, 22, 19]),
                "D": np.array([13, 16, 4, 1]),
                "L": np.array([6, 18, 12, 0]),
                "R": np.array([21, 9, 3, 15]),
                "F": np.array([20, 23, 17, 14]),
                "B": np.array([11, 8, 2, 5]),
            }

    Returns:
        marker_positions_block1 (np.ndarray, shape=(N_frames, N_markers/2, 3)): The 3D positions of all markers over N_frames for the first block.
        marker_visibility_block1 (np.ndarray, shape=(N_frames, N_markers/2)): The visibility of all markers over N_frames for the first block.
        marker_positions_block2 (np.ndarray, shape=(N_frames, N_markers/2, 3)): The 3D positions of all markers over N_frames for the second block.
        marker_visibility_block2 (np.ndarray, shape=(N_frames, N_markers/2)): The visibility of all markers over N_frames for the second block.
    """
    if designator == "U":
        facelet_index_1 = np.concatenate(
            [
                index_on_face["D"],
                index_on_face["L"][[2, 3]],
                index_on_face["F"][[2, 3]],
                index_on_face["R"][[2, 3]],
                index_on_face["B"][[2, 3]],
            ]
        )
        facelet_index_2 = np.concatenate(
            [
                index_on_face["U"],
                index_on_face["L"][[0, 1]],
                index_on_face["F"][[0, 1]],
                index_on_face["R"][[0, 1]],
                index_on_face["B"][[0, 1]],
            ]
        )
    elif designator == "R":
        facelet_index_1 = np.concatenate(
            [
                index_on_face["L"],
                index_on_face["U"][[0, 3]],
                index_on_face["F"][[0, 3]],
                index_on_face["D"][[0, 3]],
                index_on_face["B"][[1, 2]],
            ]
        )
        facelet_index_2 = np.concatenate(
            [
                index_on_face["R"],
                index_on_face["U"][[1, 2]],
                index_on_face["B"][[0, 3]],
                index_on_face["D"][[1, 2]],
                index_on_face["F"][[1, 2]],
            ]
        )
    elif designator == "F":
        facelet_index_1 = np.concatenate(
            [
                index_on_face["B"],
                index_on_face["U"][[0, 1]],
                index_on_face["R"][[1, 2]],
                index_on_face["D"][[2, 3]],
                index_on_face["L"][[0, 3]],
            ]
        )
        facelet_index_2 = np.concatenate(
            [
                index_on_face["F"],
                index_on_face["U"][[2, 3]],
                index_on_face["R"][[0, 3]],
                index_on_face["D"][[0, 1]],
                index_on_face["L"][[1, 2]],
            ]
        )
    else:
        raise ValueError(f"Invalid designator: {designator}")

    point_mask_1 = np.isin(marker_to_face_index_map, facelet_index_1)
    point_mask_2 = np.isin(marker_to_face_index_map, facelet_index_2)

    marker_positions_block1 = marker_positions[:, point_mask_1, :]
    marker_positions_block2 = marker_positions[:, point_mask_2, :]

    marker_visibility_block1 = marker_visibility[:, point_mask_1]
    marker_visibility_block2 = marker_visibility[:, point_mask_2]

    return (
        marker_positions_block1,
        marker_visibility_block1,
        marker_positions_block2,
        marker_visibility_block2,
    )


def kabsch_algorithm(
    target_marker_positions: np.ndarray,
    source_marker_positions: np.ndarray,
    visibility_target_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit the source marker positions to the target marker positions using Kabsch algorithm,
    based on the data from the first frame (index 0).

    The transformation is found such that: p_target = R @ p_source + t.

    Args:
        target_marker_positions (np.ndarray, shape=(N_markers, 3)):
            The 3D positions of all target markers.
        source_marker_positions (np.ndarray, shape=(N_markers, 3)):
            The 3D positions of all source markers. These must correspond
            to the target_marker_positions.
        visibility_target_mask (np.ndarray, shape=(N_markers,)):
            Boolean mask of target markers. True if visible, False otherwise.

    Returns:
        translation (np.ndarray, shape=(3,)):
            The translation vector t. Returns np.zeros(3) if insufficient data.
        orientation_mat (np.ndarray, shape=(3, 3)):
            The rotation matrix R. Returns np.eye(3) if insufficient data.
    """
    # Filter points based on visibility
    visible_target_pts = target_marker_positions[visibility_target_mask]
    visible_source_pts = source_marker_positions[visibility_target_mask]

    num_visible_points = visible_target_pts.shape[0]

    if num_visible_points < 3:
        # Not enough points for a unique solution
        dtype = visible_target_pts.dtype if num_visible_points > 0 else np.float64
        return np.zeros(3, dtype=dtype), np.eye(3, dtype=dtype)

    # 1. Calculate centroids
    centroid_target = np.mean(visible_target_pts, axis=0)
    centroid_source = np.mean(visible_source_pts, axis=0)

    # 2. Center the point clouds
    target_centered = visible_target_pts - centroid_target
    source_centered = visible_source_pts - centroid_source

    # 3. Calculate the covariance matrix H = source_centered.T @ target_centered
    # H should be (3xN_vis) @ (N_visx3) -> (3x3)
    H = source_centered.T @ target_centered

    # 4. Perform SVD on H
    try:
        U, S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        # SVD failed
        dtype = visible_target_pts.dtype
        return np.zeros(3, dtype=dtype), np.eye(3, dtype=dtype)

    # 5. Calculate rotation matrix R = Vt.T @ U.T
    # This rotation R aligns source_centered to target_centered: target_centered = R @ source_centered
    R = Vt.T @ U.T

    # 6. Handle reflection case: Ensure R is a proper rotation matrix (det(R) = 1)
    if np.linalg.det(R) < 0:
        # If determinant is -1, it's a reflection.
        # Multiply the column of V (Vt.T) corresponding to the smallest singular value by -1.
        V_prime = Vt.T.copy()  # V = Vt.T
        V_prime[:, -1] *= -1  # Flip the sign of the last column of V
        R = V_prime @ U.T  # Recompute R

    # 7. Calculate translation t = centroid_target - R @ centroid_source
    # Derived from: centroid_target = R @ centroid_source + t
    t = centroid_target - (R @ centroid_source)

    return t, R


def calculate_relative_rotation_between_two_122_blocks(
    twisting_designator: str,
    block1_translation: np.ndarray,
    block1_orientation_mat: np.ndarray,
    block2_translation: np.ndarray,
    block2_orientation_mat: np.ndarray,
) -> np.ndarray:
    """
    Calculate the relative rotation between two 122 blocks.

    Block1 is always D, L, B facelets.
    Block2 is always U, R, F facelets.
    Axis is from block1 to block2.

    Args:
        twisting_designator (str): The twisting designator of the 222 block.
        block1_translation (np.ndarray, shape=(3,)): The translation of the first 122 block.
        block1_orientation_mat (np.ndarray, shape=(3, 3)): The rotation matrix of the first 122 block.
        block2_translation (np.ndarray, shape=(3,)): The translation of the second 122 block.
        block2_orientation_mat (np.ndarray, shape=(3, 3)): The rotation matrix of the second 122 block.

    Returns:
        relative_rotation_angle (float): The relative rotation angle between the two 122 blocks, axis is from block1 to block2, in radians.
        mean_translation (np.ndarray, shape=(3,)): The mean translation of the two 122 blocks, will be used as the translation of the whole 222 Rubik's Cube.
        mean_orientation_mat (np.ndarray, shape=(3, 3)): The mean rotation matrix of the two 122 blocks, will be used as the rotation of the whole 222 Rubik's Cube.
    """
    R_1 = R.from_matrix(block1_orientation_mat)
    R_2 = R.from_matrix(block2_orientation_mat)
    R_relative = R_2 * R_1.inv()
    relative_rotvec = R_relative.as_rotvec(degrees=False)
    relative_rotation_angle = np.linalg.norm(relative_rotvec)
    # print(f"relative_rotvec: {R_relative.as_rotvec(degrees=True)}")
    if twisting_designator == "U":
        axis_from_block1_to_block2 = np.array([0, CUBELET_SIZE, 0])
    elif twisting_designator == "R":
        axis_from_block1_to_block2 = np.array([CUBELET_SIZE, 0, 0])
    elif twisting_designator == "F":
        axis_from_block1_to_block2 = np.array([0, 0, CUBELET_SIZE])
    else:
        raise ValueError(f"Invalid twisting designator: {twisting_designator}")
    axis_from_block1_to_block2 = R_1.apply(axis_from_block1_to_block2)
    relative_rotation_angle_sign = np.sign(
        np.dot(axis_from_block1_to_block2, relative_rotvec)
    )
    # if relative_rotation_angle > np.pi / 2:
    #     relative_rotation_angle = np.pi - relative_rotation_angle
    mean_translation = (block1_translation + block2_translation) / 2
    mean_orientation_mat = block1_orientation_mat
    return (
        -relative_rotation_angle * relative_rotation_angle_sign,
        mean_translation,
        mean_orientation_mat,
    )


def reconstruct_rubiks_animation(
    all_frames_observed_marker_positions: np.ndarray,
    all_frames_marker_visibility: np.ndarray,
    marker_to_face_index_map: np.ndarray,
    cubelet_size: float,
    gap: float,
):
    """Reconstruct the Rubik's Cube animation from the observed marker positions and visibility.

    Args:
        all_frames_observed_marker_positions (np.ndarray, shape=(N_frames, N_markers, 3)): The 3D positions of all observed markers over N_frames.
        all_frames_marker_visibility (np.ndarray, shape=(N_frames, N_markers)): The visibility of all markers over N_frames.
        marker_to_face_index_map (np.ndarray, shape=(N_markers,)): The mapping of markers to faces.
        cubelet_size (float): The size of the cubelet.
        gap (float): The gap between the cubelets.

    Returns:
        result_dict (dict): A dictionary containing the reconstructed translations, orientations, face designators, and rotation angles.
    """
    index_on_face = copy.deepcopy(INDEX_COPLANAR_ON_FACE)
    num_total_frames = all_frames_observed_marker_positions.shape[0]
    num_processed_frame = 0

    # Init result dict
    result_dict = {
        "translations": np.zeros((num_total_frames, 3), dtype=np.float32),
        "orientations": np.tile(np.eye(3, dtype=np.float32), (num_total_frames, 1, 1)),
        "face_designators": np.full((num_total_frames,), None, dtype=object),
        "rotation_angles": np.zeros((num_total_frames,), dtype=np.float32),
    }

    # Init conceptual moves
    conceptual_moves_suspected = []

    ref_simulated_data_cache = None

    # while num_processed_frame < num_total_frames:
    while (
        num_processed_frame < 8500 and num_processed_frame < num_total_frames
    ):  # TODO: random rotation can't solve, due to bad device and data
        print(f"Frame {num_processed_frame}")

        # Calculate coplanar scores for all frames
        coplanar_scores = calculate_coplanar_score(
            marker_positions=all_frames_observed_marker_positions[
                num_processed_frame : num_processed_frame + seeing_horizontal
            ],
            marker_visibility=all_frames_marker_visibility[
                num_processed_frame : num_processed_frame + seeing_horizontal
            ],
            marker_to_face_index_map=marker_to_face_index_map,
            index_coplanar_on_face=index_on_face,
        )

        # Get the twisting designator and the nearest frame offset with rotation
        twisting_designator, nearest_frame_offset_with_rotation = (
            get_twisting_designator_from_coplanar_score(coplanar_scores)
        )
        if debug_mode:
            print(f"Setting twisting_designator: {twisting_designator}")
            print(f"coplanar_scores in recent 30 frames: {coplanar_scores[:30]}")

        # If no twisting designator, means frames after this frame are not twisting.
        # But still need calculate the translation and orientation of the whole cube
        twisting_designator_is_None = False
        if twisting_designator is None:
            twisting_designator_is_None = True
            twisting_designator = "U"  # Just a placeholder

        # Separate the current frame's 222 block to 2 blocks
        (
            marker_positions_block1,
            marker_visibility_block1,
            marker_positions_block2,
            marker_visibility_block2,
        ) = separate_122_block_from_marker_positions(
            marker_positions=all_frames_observed_marker_positions,
            marker_visibility=all_frames_marker_visibility,
            marker_to_face_index_map=marker_to_face_index_map,
            designator=twisting_designator,
            index_on_face=index_on_face,
        )

        # Get reference 222 block
        if ref_simulated_data_cache is None:
            ref_simulated_data = generate_simulated_mocap_data(
                conceptual_moves=conceptual_moves_suspected,
                fps=FPS,
                cubelet_size=CUBELET_SIZE,
                gap=GAP,
                num_markers_per_side_facelet=NUM_MARKERS_PER_FACELET,
                noise_stddev=0,
                occlusion_rate=0,
                global_motion=False,
            )
            ref_simulated_data_cache = ref_simulated_data
        else:
            ref_simulated_data = ref_simulated_data_cache
        ref_marker_positions = ref_simulated_data["simulated_marker_positions_world"][
            -1:
        ]
        ref_marker_visibility = ref_simulated_data["simulated_marker_visibility"][-1:]
        ref_marker_to_face_index_map = ref_simulated_data[
            "reference_marker_canonical_face_idx"
        ]

        # Separate the reference 222 block to 2 blocks
        (
            ref_marker_positions_block1,
            ref_marker_visibility_block1,  # Useless
            ref_marker_positions_block2,
            ref_marker_visibility_block2,  # Useless
        ) = separate_122_block_from_marker_positions(
            marker_positions=ref_marker_positions,
            marker_visibility=ref_marker_visibility,
            marker_to_face_index_map=ref_marker_to_face_index_map,
            designator=twisting_designator,
            index_on_face=index_on_face,
        )

        angle_accumulated_rad = 0.0  # Until 90 or -90, stop accumulating

        twisting_designator_is_None_counter = 0
        while abs(angle_accumulated_rad) < np.pi / 2 - angle_snap_to_90_threshold_rad:
            if num_processed_frame >= num_total_frames:
                break

            # Log the two sets of four 3D points (block1 and block2) for the current frame,
            # as well as their reference positions, to Rerun for visualization.
            # Each set is logged as a point cloud with a different color for distinction.

            # Prepare the point clouds for the current frame
            block1_points = marker_positions_block1[num_processed_frame]
            block2_points = marker_positions_block2[num_processed_frame]
            ref_block1_points = ref_marker_positions_block1[0]
            ref_block2_points = ref_marker_positions_block2[0]
            block1_marker_visibility = marker_visibility_block1[num_processed_frame]
            block2_marker_visibility = marker_visibility_block2[num_processed_frame]

            if debug_mode:
                #######
                # Debug
                #######
                frame_time = num_processed_frame / FPS
                rr.set_time("frame", duration=frame_time)
                # Log block1 points (e.g., red)
                rr.log(
                    "debug/122_block1",
                    rr.Points3D(
                        block1_points[block1_marker_visibility],
                        colors=np.array([[255, 0, 0, 255]] * 4, dtype=np.uint8),
                    ),
                )
                # Log block2 points (e.g., blue)
                rr.log(
                    "debug/122_block2",
                    rr.Points3D(
                        block2_points[block2_marker_visibility],
                        colors=np.array([[0, 0, 255, 255]] * 4, dtype=np.uint8),
                    ),
                )
                # Log reference block1 points (e.g., green)
                rr.log(
                    "debug/122_block1_ref",
                    rr.Points3D(
                        ref_block1_points[block1_marker_visibility],
                        colors=np.array([[0, 255, 0, 255]] * 4, dtype=np.uint8),
                    ),
                )
                # Log reference block2 points (e.g., yellow)
                rr.log(
                    "debug/122_block2_ref",
                    rr.Points3D(
                        ref_block2_points[block2_marker_visibility],
                        colors=np.array([[255, 255, 0, 255]] * 4, dtype=np.uint8),
                    ),
                )
                #######
                # End of Debug
                #######

            # Fit the 122 block
            translation1, orientation_mat1 = kabsch_algorithm(
                block1_points,
                ref_block1_points,
                block1_marker_visibility,
            )
            translation2, orientation_mat2 = kabsch_algorithm(
                block2_points,
                ref_block2_points,
                block2_marker_visibility,
            )

            if debug_mode:
                #######
                # Debug
                #######
                # Visualize the two 122 blocks (block1 and block2) as 3D boxes in Rerun for debugging.
                # Block1 will be shown in magenta, block2 in cyan.

                # Prepare box parameters
                block_size = np.array(
                    [
                        cubelet_size * 2 + gap
                        if twisting_designator != "R"
                        else cubelet_size * 1 + gap,
                        cubelet_size * 2 + gap
                        if twisting_designator != "U"
                        else cubelet_size * 1 + gap,
                        cubelet_size * 2 + gap
                        if twisting_designator != "F"
                        else cubelet_size * 1 + gap,
                    ],
                    dtype=np.float32,
                )
                # For both blocks, the size is (2,2,1) cubelets (plus gap)

                # Block1: magenta
                rr.log(
                    "debug/122_block1_box",
                    rr.Boxes3D(
                        sizes=block_size,
                        centers=translation1
                        - np.array(
                            [
                                0 if twisting_designator != "R" else cubelet_size / 2,
                                0 if twisting_designator != "U" else cubelet_size / 2,
                                0 if twisting_designator != "F" else cubelet_size / 2,
                            ]
                        ),
                        rotations=R.from_matrix(orientation_mat1).as_quat(),
                        colors=np.array([[255, 0, 255, 255]], dtype=np.uint8),
                        fill_mode="DenseWireframe",
                    ),
                )
                # Block2: cyan
                rr.log(
                    "debug/122_block2_box",
                    rr.Boxes3D(
                        sizes=block_size,
                        centers=translation2
                        + np.array(
                            [
                                0 if twisting_designator != "R" else cubelet_size / 2,
                                0 if twisting_designator != "U" else cubelet_size / 2,
                                0 if twisting_designator != "F" else cubelet_size / 2,
                            ]
                        ),
                        rotations=R.from_matrix(orientation_mat2).as_quat(),
                        colors=np.array([[0, 255, 255, 255]], dtype=np.uint8),
                        fill_mode="DenseWireframe",
                    ),
                )
                #######
                # End of Debug
                #######

            # Calculate the relative rotation between the two 122 blocks
            relative_rotation_angle_rad, mean_translation, mean_orientation_mat = (
                calculate_relative_rotation_between_two_122_blocks(
                    twisting_designator=twisting_designator,
                    block1_translation=translation1,
                    block1_orientation_mat=orientation_mat1,
                    block2_translation=translation2,
                    block2_orientation_mat=orientation_mat2,
                )
            )
            if twisting_designator_is_None:
                relative_rotation_angle_rad = 0

            # Accumulate the rotation angle
            angle_delta_rad = relative_rotation_angle_rad - angle_accumulated_rad
            angle_accumulated_rad += angle_delta_rad

            # Update the result dict
            result_dict["translations"][num_processed_frame] = mean_translation
            result_dict["orientations"][num_processed_frame] = mean_orientation_mat
            result_dict["face_designators"][num_processed_frame] = twisting_designator
            result_dict["rotation_angles"][num_processed_frame] = angle_delta_rad

            # Increment the processed frame index
            num_processed_frame += 1

            if twisting_designator_is_None:
                twisting_designator_is_None_counter += 1
                if twisting_designator_is_None_counter >= seeing_horizontal:
                    break

        if angle_accumulated_rad > 0 and not twisting_designator_is_None:
            # print(f"angle_accumulated_rad: {angle_accumulated_rad}")
            # If the accumulated angle is greater than 90, we need to snap the angle to 90.
            result_dict["rotation_angles"][num_processed_frame - 1] -= (
                angle_accumulated_rad - np.pi / 2
            )
            # Update the index_on_face
            index_on_face = rotate_rubiks_face(
                face_designator=twisting_designator,
                index_on_face=index_on_face,
            )
            conceptual_moves_suspected.append(
                {"move_str": twisting_designator, "duration_sec": 1.0}
            )
            ref_simulated_data_cache = None
        elif angle_accumulated_rad < 0 and not twisting_designator_is_None:
            # print(f"angle_accumulated_rad: {angle_accumulated_rad}")
            # If the accumulated angle is less than -90, we need to snap the angle to -90.
            result_dict["rotation_angles"][num_processed_frame - 1] -= (
                np.pi / 2 + angle_accumulated_rad
            )
            # Update the index_on_face
            index_on_face = rotate_rubiks_face(
                face_designator=twisting_designator + "'",
                index_on_face=index_on_face,
            )
            conceptual_moves_suspected.append(
                {"move_str": twisting_designator + "'", "duration_sec": 1.0}
            )
            ref_simulated_data_cache = None

    # print(f"conceptual_moves_suspected: {conceptual_moves_suspected}")
    return result_dict


# --- Main Function for Testing ---


def main():
    """
    Main function to demonstrate Rubik's Cube reconstruction.
    1. Defines a ground truth animation sequence.
    2. Generates simulated mocap data with noise and occlusions.
    3. Attempts to reconstruct the animation from this data.
    4. Visualizes both ground truth and reconstructed animations.
    """
    print("Rubik's Cube Reconstruction Demo")

    # --- 1. Define Ground Truth Conceptual Moves ---
    # Shorter sequence for faster testing
    if VIZ_LABEL:
        conceptual_moves_gt = [
            {"move_str": "R", "duration_sec": 0.9},
        ]
    else:
        conceptual_moves_gt = [
            {"move_str": "R", "duration_sec": 0.9},
            {"move_str": "U", "duration_sec": 0.5},
            {"move_str": "R'", "duration_sec": 0.3},
            {"move_str": "U'", "duration_sec": 0.8},
            {"move_str": "R", "duration_sec": 0.9},
            {"move_str": "U", "duration_sec": 0.5},
            {"move_str": "R'", "duration_sec": 0.3},
            {"move_str": "U'", "duration_sec": 0.8},
            {"move_str": "R", "duration_sec": 0.9},
            {"move_str": "U", "duration_sec": 0.5},
            {"move_str": "R'", "duration_sec": 0.3},
            {"move_str": "U'", "duration_sec": 0.8},
            {"move_str": "R", "duration_sec": 0.9},
            {"move_str": "U", "duration_sec": 0.5},
            {"move_str": "R'", "duration_sec": 0.3},
            {"move_str": "U'", "duration_sec": 0.8},
            {"move_str": "R", "duration_sec": 0.9},
            {"move_str": "U", "duration_sec": 0.5},
            {"move_str": "R'", "duration_sec": 0.3},
            {"move_str": "U'", "duration_sec": 0.8},
            {"move_str": "R", "duration_sec": 0.9},
            {"move_str": "U", "duration_sec": 0.5},
            {"move_str": "R'", "duration_sec": 0.3},
            {"move_str": "U'", "duration_sec": 0.8},
        ]

    # # Check if the index on face is correct by `rotate_rubiks_face` function.
    # index_on_face = copy.deepcopy(INDEX_ON_FACE)
    # print("Initial index on face:")
    # print(index_on_face)
    # for move in conceptual_moves_gt:
    #     print(f'After {move["move_str"]}:')
    #     index_on_face = rotate_rubiks_face(move["move_str"], index_on_face)
    #     print(f"{index_on_face}")

    # --- 2. Generate Simulated Mocap Data ---
    simulated_data = generate_simulated_mocap_data(
        conceptual_moves=conceptual_moves_gt,
        fps=FPS,
        cubelet_size=CUBELET_SIZE,
        gap=GAP,
        num_markers_per_side_facelet=NUM_MARKERS_PER_FACELET,
        noise_stddev=NOISE_STD,
        occlusion_rate=OCCLUSION_PROB,
    )
    total_frames_sim = simulated_data["total_frames"]
    sim_marker_positions_world = simulated_data["simulated_marker_positions_world"]
    sim_marker_visibility = simulated_data["simulated_marker_visibility"]
    model_marker_to_canonical_face_index_map = simulated_data[
        "reference_marker_canonical_face_idx"
    ]
    fps_for_viz = simulated_data.get("fps", FPS)  # Use FPS from simulation data

    if use_real_data:
        sim_marker_positions_world = np.load(mocap_points_path)
        sim_marker_positions_world = sim_marker_positions_world[
            num_frames_start:num_frames_end
        ]

        model_marker_to_canonical_face_index_map = np.load(facelet_mask_path)  # 0-23
        each_patch_point_index = np.load(patch_mask_path)  # 0-15

        # Change point sequence to `each_patch_point_index`
        new_sim_marker_positions_world = np.zeros_like(sim_marker_positions_world)
        for i in range(0, 24 * 16, 16):
            new_index = model_marker_to_canonical_face_index_map[i] * 16
            new_sim_marker_positions_world[:, new_index : new_index + 16, :] = (
                sim_marker_positions_world[
                    :,
                    i + each_patch_point_index[i : i + 16],
                    :,
                ].copy()
            )
        sim_marker_positions_world = new_sim_marker_positions_world.copy()

        model_marker_to_canonical_face_index_map = sorted(
            model_marker_to_canonical_face_index_map
        )

        sim_marker_visibility = ~np.all(
            np.isclose(sim_marker_positions_world, [-1000, -1000, -1000]), axis=-1
        )
        total_frames_sim = sim_marker_positions_world.shape[0]
        fps_for_viz = 20

    # --- 3. Reconstruct Animation ---
    # This will currently only reconstruct global pose.
    # Internal face rotations are not yet implemented in reconstruct_rubiks_animation.

    if debug_mode:
        RERUN_APP_NAME_COMBINED = "RubiksCube_Reconstruction_Debug"
        init_rerun(RERUN_APP_NAME_COMBINED, save=False)

    print("Reconstructing animation...")
    reconstructed_animation_data = reconstruct_rubiks_animation(
        all_frames_observed_marker_positions=sim_marker_positions_world,
        all_frames_marker_visibility=sim_marker_visibility,
        marker_to_face_index_map=model_marker_to_canonical_face_index_map,
        cubelet_size=CUBELET_SIZE,
        gap=GAP,
    )

    # Save the reconstructed animation data as a .npz file for later use.
    # The file will contain translations, orientations, face_designators, and rotation_angles.
    if save_reconstructed_data:
        np.savez(
            os.path.join(
                os.path.dirname(__file__),
                "Result",
                "RubiksCube_00",
                "2025_05_20_05_01_31",
                f"reconstructed_rubikscube_data-{num_frames_start}-{num_frames_end}.npz",
            ),
            translations=reconstructed_animation_data["translations"],
            orientations=reconstructed_animation_data["orientations"],
            face_designators=reconstructed_animation_data["face_designators"],
            rotation_angles=reconstructed_animation_data["rotation_angles"],
        )
        exit()

    if debug_mode:
        print("Reconstructed animation data:")
        face_designators = reconstructed_animation_data["face_designators"]
        rotation_angles = reconstructed_animation_data["rotation_angles"]
        accumulated_rotation_angles = 0
        for i in range(len(face_designators)):
            if i != 0 and face_designators[i] != face_designators[i - 1]:
                accumulated_rotation_angles = 0
            accumulated_rotation_angles += rotation_angles[i]
            print(f"Frame {i + num_frames_start}:")
            print(
                f"Reconstructed: {face_designators[i]}, {rotation_angles[i] / np.pi * 180} deg, Accumulated: {accumulated_rotation_angles / np.pi * 180} deg"
            )
            # print(
            #     f"translation: {reconstructed_animation_data['translations'][i]}, orientation: {reconstructed_animation_data['orientations'][i]}"
            # )
            # print(
            #     f"GT: {simulated_data['ground_truth_animation_data']['face_designators'][i]}, {simulated_data['ground_truth_animation_data']['rotation_angles'][i] / np.pi * 180} deg"
            # )

    # --- 4. Visualization: Combined Mocap Markers and Reconstructed Cube ---
    RERUN_APP_NAME_COMBINED = "RubiksCube_Reconstruction_Demo"
    init_rerun(RERUN_APP_NAME_COMBINED, save=False)

    # Step 1: Visualize the reconstructed cube.
    # visualize_rubiks_cube_animation will call init_rerun internally for RERUN_APP_NAME_COMBINED.
    # This should establish the Rerun session and server.

    print(
        f"\n--- Initializing Rerun and Logging Reconstructed Cube to '{RERUN_APP_NAME_COMBINED}' ---"
    )
    visualize_rubiks_cube_animation(
        animation_data=reconstructed_animation_data,
        animation_fps=fps_for_viz,
        cubelet_size=CUBELET_SIZE,
        gap=GAP,
        cube_name="RubiksCube_Reconstructed",
    )

    # Step 2: Log Mocap Markers to the same, now active, Rerun session.
    # Set view coordinates for the "world" path if not already set by visualize_rubiks_cube_animation.
    # This is useful if mocap_points are logged directly under "world/mocap_points".

    log_mocap_points(
        sim_marker_positions_world=sim_marker_positions_world,
        sim_marker_visibility=sim_marker_visibility,
        model_marker_to_canonical_face_index_map=model_marker_to_canonical_face_index_map,
        total_frames=total_frames_sim,
        fps=fps_for_viz,
        viz_label=VIZ_LABEL,
    )

    # Step 3: Ground Truth visualization - this will initialize its own separate Rerun instance.
    if use_real_data:
        exit()
    gt_viz_data = simulated_data["ground_truth_animation_data"].copy()
    gt_viz_data["translations"] += np.array([0, 0, CUBELET_SIZE * 4])
    visualize_rubiks_cube_animation(
        animation_data=gt_viz_data,
        animation_fps=fps_for_viz,
        cubelet_size=CUBELET_SIZE,
        gap=GAP,
        cube_name="RubiksCube_GT",
    )


if __name__ == "__main__":
    main()
