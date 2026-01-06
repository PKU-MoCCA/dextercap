import argparse
import copy
import glob
import importlib.util
import os
from typing import Any, Dict

import numpy as np
import smplx
import torch
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm

MANO_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    os.path.pardir,
    "HandReconstruction",
    "Data",
    "HumanModels",
)


def get_motion_object_data(mocap_name, hand_param_folder, object_param_folder):
    """
    Read all npz files in the folder, connect them, and segment based on index_init_frame in config

    Parameters:
    - mocap_name: name of the mocap session, use for checking the consistency of the data
    - hand_param_folder: folder path containing npz files
    - object_param_folder: folder path containing npz files
    """
    ##########
    # Hand data
    ##########

    # Get all npz files
    npz_files = sorted(glob.glob(os.path.join(hand_param_folder, "*.npz")))

    if not npz_files:
        print(f"No npz files found in {hand_param_folder}")
        return

    print(f"Found {len(npz_files)} npz files for hand data")

    # Initialize a dictionary to store connected data
    hand_data = {}

    # Read the first file to get the data structure
    first_data = np.load(npz_files[0])
    for key in first_data.keys():
        hand_data[key] = []

    # Read all files and connect data
    for npz_file in npz_files:
        data = np.load(npz_file)
        for key in hand_data.keys():
            if key in data:
                hand_data[key].append(data[key])

    # Connect all data
    for key in hand_data.keys():
        hand_data[key] = np.concatenate(hand_data[key], axis=0)
        print(f"Shape of raw data {key}: {hand_data[key].shape}")

    # Get the config file path
    # This part of the code is responsible for loading a Python module from a specified file path. Here's
    # a breakdown of what each step does:
    hand_config_file_path = os.path.normpath(
        os.path.join(hand_param_folder, os.pardir, "config.py")
    )
    spec = importlib.util.spec_from_file_location("hand_config", hand_config_file_path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load hand config from {hand_config_file_path}")
    hand_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hand_config)
    assert hand_config.mocap_session_name == mocap_name, (
        "Mocap session name mismatch: "
        f"    hand_config: {hand_config.mocap_session_name} != specified mocap_name: {mocap_name}"
    )

    # Get segment indices
    index_init_frame = hand_config.optimize["index_init_frame"]
    print(f"Data initial frames: {index_init_frame}")

    ##########
    # Object data
    ##########

    use_rubiks_cube_data = False
    if "RubiksCube" in mocap_name:
        use_rubiks_cube_data = True

    # Get object data
    file_list = sorted(
        glob.glob(
            os.path.join(
                object_param_folder, "*.npy" if not use_rubiks_cube_data else "*.npz"
            )
        )
    )
    assert len(file_list) == 1, "Only one npy file is allowed in the folder"
    npy_file_path = file_list[0]

    if not use_rubiks_cube_data:
        object_data = np.load(npy_file_path)
    else:
        object_data = np.load(npy_file_path, allow_pickle=True)

    if not use_rubiks_cube_data:
        print(f"Shape of raw object data: {object_data.shape}")
    else:
        print(f"Shape of raw object data: {object_data['translations'].shape}")

    # Get the config file path
    # This part of the code is responsible for loading a Python module from a specified file path. Here's
    # a breakdown of what each step does:
    object_config_file_path = os.path.normpath(
        os.path.join(object_param_folder, os.pardir, "config.py")
    )
    spec = importlib.util.spec_from_file_location(
        "object_config", object_config_file_path
    )
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load object config from {object_config_file_path}")
    object_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(object_config)
    assert object_config.mocap_session_name == mocap_name, (
        "Mocap session name mismatch: "
        f"    object_config: {object_config.mocap_session_name} != specified mocap_name: {mocap_name}"
    )

    # Get the object data range
    object_data_range = np.array(object_config.data["object_frame_range"])  # [885, 920]
    if object_data_range[1] == -1:
        object_data_range[1] = hand_data["optimized_translations"].shape[0]
    print(f"Object data range relative to the hand data: {object_data_range}")

    ##########
    # Put together
    ##########

    # Check if the length of the object data is the same as described in the `object_data_range`
    if not use_rubiks_cube_data:
        expected_length_object_data = hand_data["optimized_translations"][
            object_data_range[0] : object_data_range[1]
        ].shape[0]

        if object_data.shape[0] == expected_length_object_data + 1:
            object_data = object_data[
                :-1
            ]  # TODO: This is a temporary fix for the object data len mismatch

        length_object_data = object_data.shape[0]
        assert length_object_data == expected_length_object_data, (
            f"object_data_range: {object_data_range} is not consistent with the hand data"
            "Object data length mismatch: "
            f"    object_data: {length_object_data} != expected_length_object_data: {expected_length_object_data}"
        )

        # Extend the object data to the same length as the hand data
        object_data_padding = np.array([-1000.0, -1000.0, -1000.0, 0.0, 0.0, 0.0, 1.0])
        len_hand_data = hand_data["optimized_translations"].shape[0]
        object_data_padding = np.tile(object_data_padding, (len_hand_data, 1))
        object_data_padding[object_data_range[0] : object_data_range[1]] = object_data
    else:
        object_orientations_quat_xyzw = R.from_matrix(
            object_data["orientations"]
        ).as_quat()
        object_translations = object_data["translations"]
        object_face_designators = object_data["face_designators"]
        object_rotation_angles = object_data["rotation_angles"]

    # Segment hand data
    data = {}
    data["metadata"] = {}
    data["metadata"]["description_string"] = """
    This data is about the hand and object.
    - metadata:
        - mocap_session_name: name of the mocap session, e.g. "Cuboid_00"
        - index_init_frame: index of the initial frame that separates 2 sequences, which frames are the frames where the hand reappears, indicating that this frame and the previous frame are two different action sequences. If all sequences are continuous, then index_init_frame = [0]. If the hand comes in from outside the camera in the tenth frame, then index_init_frame = [0, 10].
        - invalid_point_value: value of the invalid point, e.g. [-1000, -1000, -1000] in the `object_translations` stands for the object is not present
        - object_class: class of the object, e.g. "Cuboid"
        - object_size: size of the object, if the object is a cuboid, then the size is the edge length of the cuboid
    - hand_translations: translations of the hand
    - hand_rotations_axis_angle: rotations of the hand
    - hand_poses: poses of the hand
    - hand_shapes: shapes of the hand
    - object_translations: translations of the object
    - object_rotations_quat_xyzw: rotations of the object
    """
    data["metadata"]["mocap_session_name"] = mocap_name
    data["metadata"]["index_init_frame"] = index_init_frame
    data["metadata"]["invalid_point_value"] = np.array([-1000.0, -1000.0, -1000.0])
    data["metadata"]["object_class"] = mocap_name.split("_")[0]
    data["metadata"]["object_size"] = object_config.data["object_size"]
    data["metadata"]["fps"] = 20

    data["hand_translations"] = hand_data["optimized_translations"]
    data["hand_orientations_axis_angle"] = hand_data["optimized_orientations"]
    data["hand_poses"] = hand_data["optimized_poses"]
    data["hand_shapes"] = hand_data["optimized_shapes"]

    if not use_rubiks_cube_data:
        data["object_translations"] = object_data_padding[:, :3]
        data["object_orientations_quat_xyzw"] = object_data_padding[:, 3:]
    else:
        data["object_translations"] = object_translations
        data["object_orientations_quat_xyzw"] = object_orientations_quat_xyzw
        data["object_face_designators"] = object_face_designators
        data["object_rotation_angles"] = object_rotation_angles

    return data


def change_mano_translation_to_root_translation(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Change the translation of the hand (defined by the MANO model) to the root translation.

    We use Kabsch Algorithm to find the best orientation and translation.
    """
    num_frames_hand = data["hand_translations"].shape[0]

    smplx_model = smplx.create(
        model_path=MANO_MODEL_PATH,
        model_type="mano",
        flat_hand_mean=True,
        is_rhand=False,
        use_pca=False,
        batch_size=num_frames_hand,
    )

    hand_parms = {
        "global_orient": torch.tensor(
            data["hand_orientations_axis_angle"].reshape(num_frames_hand, 3),
            dtype=torch.float32,
        ),
        "transl": torch.tensor(
            data["hand_translations"].reshape(num_frames_hand, 3), dtype=torch.float32
        ),
        "hand_pose": torch.tensor(
            data["hand_poses"].reshape(num_frames_hand, 45), dtype=torch.float32
        ),
        "betas": torch.tensor(
            data["hand_shapes"].reshape(num_frames_hand, 10), dtype=torch.float32
        ),
    }
    smplx_output = smplx_model(**hand_parms)
    target_hand_vertices = smplx_output.vertices.detach().cpu().numpy()

    hand_parms = {
        "global_orient": torch.zeros((num_frames_hand, 3), dtype=torch.float32),
        "transl": torch.zeros((num_frames_hand, 3), dtype=torch.float32),
        "hand_pose": torch.tensor(
            data["hand_poses"].reshape(num_frames_hand, 45), dtype=torch.float32
        ),
        "betas": torch.tensor(
            data["hand_shapes"].reshape(num_frames_hand, 10), dtype=torch.float32
        ),
    }
    smplx_output = smplx_model(**hand_parms)
    source_hand_vertices = smplx_output.vertices.detach().cpu().numpy()

    # Use Kabsch algorithm to find the best orientation and translation that minimizes the distance between the target and source hand vertices
    hand_translations_from_kabsch_list = []
    hand_orientations_axis_angle_from_kabsch_list = []

    for i in tqdm(
        range(num_frames_hand), desc="Changing MANO translation to root translation"
    ):
        P_frame = source_hand_vertices[i]  # (num_vertices, 3)
        Q_frame = target_hand_vertices[i]  # (num_vertices, 3)

        # 1. Center point clouds
        centroid_P = np.mean(P_frame, axis=0)
        centroid_Q = np.mean(Q_frame, axis=0)
        P_centered = P_frame - centroid_P
        Q_centered = Q_frame - centroid_Q

        # 2. Covariance matrix H
        H = P_centered.T @ Q_centered  # (3, num_vertices) @ (num_vertices, 3) -> (3, 3)

        # 3. SVD
        U, S, Vt = np.linalg.svd(H)

        # 4. Rotation matrix R
        R_optimal = Vt.T @ U.T

        # Ensure a right-handed coordinate system (no reflection)
        if np.linalg.det(R_optimal) < 0:
            Vt_corrected = Vt.copy()
            Vt_corrected[2, :] *= -1
            R_optimal = Vt_corrected.T @ U.T

        # 5. Translation vector t
        t_optimal = centroid_Q - R_optimal @ centroid_P

        hand_translations_from_kabsch_list.append(t_optimal)
        hand_orientations_axis_angle_from_kabsch_list.append(
            R.from_matrix(R_optimal).as_rotvec()
        )

    hand_translations_from_kabsch = np.array(hand_translations_from_kabsch_list)
    hand_orientations_axis_angle_from_kabsch = np.array(
        hand_orientations_axis_angle_from_kabsch_list
    )

    new_data = copy.deepcopy(data)
    new_data["hand_translations"] = hand_translations_from_kabsch
    new_data["hand_orientations_axis_angle"] = hand_orientations_axis_angle_from_kabsch
    return new_data


def interpolate_translations(
    data: np.ndarray, old_fps: int, new_fps: int
) -> np.ndarray:
    """Interpolate translations to a target framerate using cubic or linear interpolation.

    - Uses cubic interpolation if number of frames >= 4.
    - Uses linear interpolation if number of frames is 2 or 3.
    - Repeats the frame if number of frames is 1.
    - Returns an empty array if input data is empty.

    Args:
        data (np.ndarray, shape=(N, D)): Input translation data, N frames, D dimensions.
        old_fps (int): Original framerate of the data.
        new_fps (int): Target framerate.

    Returns:
        np.ndarray, shape=(M, D): Interpolated translation data, M frames.
    """
    n_frames = data.shape[0]
    if n_frames == 0:
        return np.empty((0, data.shape[1] if data.ndim == 2 else 0), dtype=data.dtype)

    # Calculate the number of frames for the new framerate
    # Use round to get the closest integer number of frames
    new_n_frames = int(round(n_frames * new_fps / old_fps))
    if (
        new_n_frames == 0 and n_frames > 0
    ):  # Ensure at least one frame if original was not empty
        new_n_frames = 1

    if n_frames == 1:
        return np.tile(data, (new_n_frames, 1))

    # Create time points based on frame indices
    old_times = np.linspace(0, n_frames - 1, n_frames)
    new_times = np.linspace(0, n_frames - 1, new_n_frames)

    if n_frames >= 4:
        kind = "cubic"
    else:  # 2 or 3 frames
        kind = "linear"

    interpolator = interp1d(
        old_times, data, axis=0, kind=kind, fill_value="extrapolate"
    )
    return interpolator(new_times)


def interpolate_rotations(data: np.ndarray, old_fps: int, new_fps: int) -> np.ndarray:
    """Interpolate rotations (represented as rotation vectors) to a target framerate using SLERP.

    - Uses SLERP if number of frames >= 2.
    - Repeats the frame if number of frames is 1.
    - Returns an empty array if input data is empty.

    Args:
        data (np.ndarray, shape=(N, 3)): Input rotation vectors.
        old_fps (int): Original framerate of the data.
        new_fps (int): Target framerate.

    Returns:
        np.ndarray, shape=(M, 3): Interpolated rotation vectors.
    """
    n_frames = data.shape[0]
    if n_frames == 0:
        return np.empty((0, 3), dtype=data.dtype)  # Assuming 3 for rotvecs

    # Calculate the number of frames for the new framerate
    new_n_frames = int(round(n_frames * new_fps / old_fps))
    if (
        new_n_frames == 0 and n_frames > 0
    ):  # Ensure at least one frame if original was not empty
        new_n_frames = 1

    if n_frames == 1:
        return np.tile(data, (new_n_frames, 1))

    # Convert to rotation objects
    rots = R.from_rotvec(data)

    # Create time points based on frame indices
    old_times = np.linspace(0, n_frames - 1, n_frames)
    new_times = np.linspace(0, n_frames - 1, new_n_frames)

    # Create Slerp interpolator
    slerp = Slerp(old_times, rots)

    # Interpolate and convert back to rotvec
    return slerp(new_times).as_rotvec()


def smooth_translations(
    data: np.ndarray, cutoff: float = 6.0, fs: float = 30.0, order: int = 2
) -> np.ndarray:
    """
    Smooth translations using a Butterworth low-pass filter

    Parameters:
    - data: Input translations (N x 3)
    - cutoff: Cutoff frequency of the filter in Hz
    - fs: Sampling frequency in Hz
    - order: Order of the filter
    """
    # Design the Butterworth filter
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    # Apply filter to each dimension
    smoothed = np.zeros_like(data)
    for i in range(3):
        smoothed[:, i] = filtfilt(b, a, data[:, i])

    return smoothed


def smooth_rotations(
    data: np.ndarray, cutoff: float = 6.0, fs: float = 30.0, order: int = 2
) -> np.ndarray:
    """
    Smooth rotations using a Butterworth low-pass filter applied to quaternions.

    This method converts rotation vectors to quaternions, ensures quaternion continuity
    (shortest path), filters the quaternion components, normalizes the result, and
    converts back to rotation vectors. This approach avoids singularity issues
    inherent in filtering rotation vectors directly, especially near 180-degree rotations.

    Parameters:
    - data (np.ndarray, shape=(N, 3)): Input array of N rotation vectors. Each vector's
        direction is the axis of rotation and magnitude is the angle in radians.
    - cutoff (float): Cutoff frequency of the filter in Hz. Frequencies above this
        will be attenuated. Defaults to 6.0.
    - fs (float): Sampling frequency of the input data in Hz. Defaults to 30.0.
    - order (int): Order of the Butterworth filter. Higher orders provide a steeper
        rolloff but can introduce ringing. Defaults to 2.

    Returns:
    - np.ndarray, shape=(N, 3): Smoothed rotation vectors.
    """
    # Design the Butterworth filter
    nyq = 0.5 * fs
    if cutoff >= nyq:
        print(
            f"Warning: Cutoff frequency ({cutoff} Hz) is >= Nyquist frequency ({nyq} Hz). "
            f"Filter will not be effective. Returning original data."
        )
        return (
            data.copy()
        )  # Return a copy to avoid modifying original data if filtering is skipped
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    # 1. Convert rotation vectors to quaternions
    rotations = R.from_rotvec(data)
    quaternions = rotations.as_quat()  # Shape (N, 4), format [x, y, z, w]

    # 2. Ensure quaternion continuity (shortest path interpolation)
    # Make sure we always take the shortest path between consecutive orientations
    continuous_quats = np.copy(quaternions)
    for i in range(1, len(continuous_quats)):
        # Dot product between consecutive quaternions
        dot_product = np.dot(continuous_quats[i - 1], continuous_quats[i])
        # If dot product is negative, the quaternions are pointing in opposite directions
        # (representing the same rotation but via the "long path"). Flip the sign of the current quaternion.
        if dot_product < 0.0:
            continuous_quats[i] *= -1.0

    # 3. Filter quaternion components
    smoothed_quats = np.zeros_like(continuous_quats)
    for i in range(4):  # Filter x, y, z, w components
        # Apply zero-phase filter to avoid phase distortion
        smoothed_quats[:, i] = filtfilt(b, a, continuous_quats[:, i])

    # 4. Normalize the filtered quaternions to ensure they remain valid rotations
    norms = np.linalg.norm(smoothed_quats, axis=1, keepdims=True)
    # Avoid division by zero if a quaternion becomes zero (unlikely but possible)
    # If norm is close to zero, set norm to 1 to avoid NaN; the resulting quat will be arbitrary but won't crash.
    # A zero quat often implies near-zero rotation anyway.
    zero_norms_mask = norms < 1e-8
    norms[zero_norms_mask] = 1.0
    normalized_quats = smoothed_quats / norms

    # 5. Convert smoothed, normalized quaternions back to rotation vectors
    smoothed_rotations = R.from_quat(normalized_quats)
    smoothed_rotvecs = smoothed_rotations.as_rotvec()

    return smoothed_rotvecs


def interpolate_strings(
    data: np.ndarray, old_fps: int, new_fps: int, target_n_frames: int
) -> np.ndarray:
    """Interpolate string data using nearest neighbor repetition.

    Each original frame's value is repeated to fill the corresponding
    time slot in the upsampled sequence.

    Args:
        data (np.ndarray, shape=(N,)): Input string data.
        old_fps (int): Original framerate of the data.
        new_fps (int): Target framerate.
        target_n_frames (int): The exact number of frames for the output array.

    Returns:
        np.ndarray, shape=(target_n_frames,): Interpolated string data.
    """
    n_frames_old = data.shape[0]

    if target_n_frames == 0:
        return np.array([], dtype=data.dtype if n_frames_old > 0 else object)

    if n_frames_old == 0:  # Input data is empty
        # The caller (smooth_data) handles filling if target_n_frames > 0
        return np.array([], dtype=object)

    if n_frames_old == 1:  # Single data point to be tiled
        return np.full(target_n_frames, data[0], dtype=data.dtype)

    interpolated_data = np.empty(target_n_frames, dtype=data.dtype)
    for i_new in range(target_n_frames):
        # Map new frame index to old frame index
        # Each old frame k is valid for time t_k to t_{k+1}
        # A new frame at time t_new should take data from old frame k if t_new is in [t_k, t_{k+1})
        # Equivalent to: old_idx = floor(new_idx * (n_frames_old / target_n_frames))
        pos_old = i_new * (n_frames_old / target_n_frames)
        i_old = int(np.floor(pos_old))
        i_old = min(i_old, n_frames_old - 1)  # Clamp to last valid index
        interpolated_data[i_new] = data[i_old]

    return interpolated_data


def smooth_data(
    data_dict: Dict[str, Any],
    cutoff: float = 5.0,
    old_fps: int = 20,
    new_fps: int = 60,
    filter_order: int = 2,
) -> Dict[str, Any]:
    """
    Smooth motion data segment by segment, then upsample to a higher framerate.

    Segments are defined by `data_dict['metadata']['index_init_frame']`.
    Smoothing and upsampling do not cross segment boundaries.

    For object data, frames with `invalid_point_value` are not smoothed but are
    upsampled by repeating the invalid value to maintain frame consistency.

    Args:
        data_dict (Dict[str, Any]): Dictionary containing motion data and metadata.
            Expected keys include 'metadata', 'hand_translations', 'hand_rotations_axis_angle',
            'hand_poses', 'hand_shapes', 'object_translations', 'object_rotations_quat_xyzw'.
        cutoff (float): Cutoff frequency for the Butterworth low-pass filter in Hz.
        old_fps (int): Original framerate of the input data.
        new_fps (int): Target framerate for upsampling.
        filter_order (int): Order of the Butterworth filter.

    Returns:
        Dict[str, Any]: A new dictionary with smoothed and upsampled data, and updated metadata.
    """
    processed_data = {"metadata": data_dict["metadata"].copy()}  # type: Dict[str, Any]
    # N > 3 * (order + 1) for filtfilt padlen, so N must be at least 3*(order+1) + 1
    min_len_for_smooth = 3 * (filter_order + 1) + 1

    mocap_name = data_dict["metadata"].get("mocap_session_name", "")
    use_rubiks_cube_data = "RubiksCube" in mocap_name

    # Determine original total number of frames from available data
    original_total_frames = 0
    keys_to_check_len = [
        "hand_translations",
        "hand_orientations_axis_angle",
        "hand_poses",
        "object_translations",
        "object_orientations_quat_xyzw",
        "hand_shapes",
    ]
    if use_rubiks_cube_data:
        keys_to_check_len.extend(["object_rotation_angles", "object_face_designators"])

    for key_to_check_len in keys_to_check_len:
        if (
            key_to_check_len in data_dict
            and data_dict[key_to_check_len] is not None
            and len(data_dict[key_to_check_len]) > 0
        ):
            original_total_frames = len(data_dict[key_to_check_len])
            break

    keys_to_process = [
        "hand_translations",
        "hand_orientations_axis_angle",
        "hand_poses",
        "hand_shapes",
        "object_translations",
        "object_orientations_quat_xyzw",
    ]
    if use_rubiks_cube_data:
        keys_to_process.extend(["object_rotation_angles", "object_face_designators"])

    if original_total_frames == 0:
        print("Warning: No data frames to process in smooth_data.")
        # Default dtypes and whether they are 1D (None for shape_dim means 1D)
        # (shape_dim, dtype)
        key_configs_on_empty = {
            "hand_translations": (3, float),
            "hand_orientations_axis_angle": (3, float),
            "hand_poses": (45, float),
            "hand_shapes": (10, float),
            "object_translations": (3, float),
            "object_orientations_quat_xyzw": (4, float),
            "object_rotation_angles": (None, float),  # 1D, results in (0,)
            "object_face_designators": (None, object),  # 1D, results in (0,)
        }

        for key in keys_to_process:
            if key in data_dict and data_dict[key] is not None:
                # Try to mimic original structure if available
                original_arr = data_dict[key]
                empty_shape = list(original_arr.shape)
                empty_shape[0] = 0
                processed_data[key] = np.empty(
                    tuple(empty_shape), dtype=original_arr.dtype
                )
            elif key in key_configs_on_empty:
                shape_dim, dtype = key_configs_on_empty[key]
                if shape_dim is not None:  # It's 2D, create (0, D)
                    processed_data[key] = np.empty((0, shape_dim), dtype=dtype)
                else:  # It's 1D, create (0,)
                    processed_data[key] = np.array([], dtype=dtype)
            # else: A key in keys_to_process is not in data_dict and not in key_configs_on_empty.
            # This implies an oversight in key_configs_on_empty or keys_to_process list.
            # For robustness, ensure all keys_to_process are initialized if they reach here.
            elif key not in processed_data:  # If not yet initialized by above branches
                print(
                    f"Warning: Key {key} not in data_dict or key_configs_on_empty, initializing as generic empty array."
                )
                processed_data[key] = np.array([])

        processed_data["metadata"]["index_init_frame"] = [
            0
        ]  # Convention for empty sequence
        return processed_data

    original_indices = processed_data["metadata"].get("index_init_frame", [0])
    if isinstance(original_indices, np.ndarray):
        original_indices = original_indices.tolist()
    if not original_indices:  # If empty list, assume one segment from frame 0
        original_indices = [0]

    iter_boundaries = []
    current_indices = sorted(list(set(original_indices)))  # Unique sorted start frames
    for i in range(len(current_indices)):
        start_idx = current_indices[i]
        end_idx = (
            current_indices[i + 1]
            if i + 1 < len(current_indices)
            else original_total_frames
        )
        if start_idx < end_idx:  # Ensure segment has non-zero length
            iter_boundaries.append((start_idx, end_idx))

    if not iter_boundaries and original_total_frames > 0:  # Single segment [0, N]
        iter_boundaries.append((0, original_total_frames))

    all_processed_segments = {k: [] for k in keys_to_process}

    new_segment_start_frames = [0]
    cumulative_upsampled_frames = 0

    invalid_point_value = np.array(
        processed_data["metadata"].get(
            "invalid_point_value", [-1000.0, -1000.0, -1000.0]
        )
    )
    identity_quaternion = np.array([0.0, 0.0, 0.0, 1.0])  # xyzw

    print(f"iter_boundaries for smooth and upsample: {iter_boundaries}")
    for seg_idx, (start_idx, end_idx) in enumerate(iter_boundaries):
        len_seg = end_idx - start_idx
        num_upsampled_frames_seg = 0
        if len_seg > 0:
            num_upsampled_frames_seg = int(round(len_seg * new_fps / old_fps))
            if (
                num_upsampled_frames_seg == 0
            ):  # Ensure at least one frame if original segment not empty
                num_upsampled_frames_seg = 1

        # --- Hand Data ---
        if "hand_translations" in data_dict:
            seg_data = data_dict["hand_translations"][start_idx:end_idx]
            if len_seg >= min_len_for_smooth:
                seg_data = smooth_translations(
                    seg_data, cutoff=cutoff, fs=old_fps, order=filter_order
                )
            upsampled_seg = interpolate_translations(seg_data, old_fps, new_fps)
            all_processed_segments["hand_translations"].append(upsampled_seg)

        if "hand_orientations_axis_angle" in data_dict:
            seg_data = data_dict["hand_orientations_axis_angle"][start_idx:end_idx]
            if len_seg >= min_len_for_smooth:
                seg_data = smooth_rotations(
                    seg_data, cutoff=cutoff, fs=old_fps, order=filter_order
                )
            upsampled_seg = interpolate_rotations(seg_data, old_fps, new_fps)
            all_processed_segments["hand_orientations_axis_angle"].append(upsampled_seg)

        if "hand_poses" in data_dict:
            seg_data_poses = data_dict["hand_poses"][start_idx:end_idx]
            if seg_data_poses.shape[0] > 0:  # only process if there are frames
                num_frames_pose, D_pose = seg_data_poses.shape
                seg_data_poses_reshaped = seg_data_poses.reshape(num_frames_pose, 15, 3)
                smoothed_poses_joints = np.copy(seg_data_poses_reshaped)
                if len_seg >= min_len_for_smooth:
                    for joint_idx in range(15):
                        smoothed_poses_joints[:, joint_idx, :] = smooth_rotations(
                            seg_data_poses_reshaped[:, joint_idx, :],
                            cutoff=cutoff,
                            fs=old_fps,
                            order=filter_order,
                        )

                upsampled_poses_joints = np.zeros(
                    (num_upsampled_frames_seg, 15, 3), dtype=smoothed_poses_joints.dtype
                )
                for joint_idx in range(15):
                    upsampled_poses_joints[:, joint_idx, :] = interpolate_rotations(
                        smoothed_poses_joints[:, joint_idx, :], old_fps, new_fps
                    )
                all_processed_segments["hand_poses"].append(
                    upsampled_poses_joints.reshape(num_upsampled_frames_seg, D_pose)
                )
            else:
                all_processed_segments["hand_poses"].append(
                    np.empty(
                        (0, data_dict["hand_poses"].shape[1]),
                        dtype=data_dict["hand_poses"].dtype,
                    )
                )

        if "hand_shapes" in data_dict:
            if (
                len_seg > 0 and data_dict["hand_shapes"].shape[0] > start_idx
            ):  # Ensure there is a shape to pick
                # Tile the shape from the first frame of the original segment
                shape_to_tile = data_dict["hand_shapes"][start_idx : start_idx + 1]
                upsampled_shapes = np.tile(shape_to_tile, (num_upsampled_frames_seg, 1))
                all_processed_segments["hand_shapes"].append(upsampled_shapes)
            elif num_upsampled_frames_seg > 0:  # No original shape, but frames expected
                all_processed_segments["hand_shapes"].append(
                    np.zeros(
                        (
                            num_upsampled_frames_seg,
                            data_dict["hand_shapes"].shape[1]
                            if data_dict["hand_shapes"].ndim > 1
                            else 0,
                        ),  # handle 0-dim shape[1]
                        dtype=data_dict["hand_shapes"].dtype,
                    )
                )
            else:  # No original shape and no upsampled frames expected
                all_processed_segments["hand_shapes"].append(
                    np.empty(
                        (0, data_dict["hand_shapes"].shape[1]),
                        dtype=data_dict["hand_shapes"].dtype,
                    )
                )

        # --- Object Data --- Ensure consistent frame count for this segment
        # Use length of upsampled hand translations if available, else calculated num_upsampled_frames_seg
        current_segment_upsampled_len = num_upsampled_frames_seg
        if (
            all_processed_segments["hand_translations"]
            and len(all_processed_segments["hand_translations"][-1]) > 0
        ):
            current_segment_upsampled_len = len(
                all_processed_segments["hand_translations"][-1]
            )
        elif (
            all_processed_segments["hand_orientations_axis_angle"]
            and len(all_processed_segments["hand_orientations_axis_angle"][-1]) > 0
        ):
            current_segment_upsampled_len = len(
                all_processed_segments["hand_orientations_axis_angle"][-1]
            )

        if "object_translations" in data_dict:
            obj_trans_seg = data_dict["object_translations"][start_idx:end_idx]
            upsampled_obj_trans = np.full(
                (
                    current_segment_upsampled_len,
                    obj_trans_seg.shape[1] if obj_trans_seg.ndim > 1 else 3,
                ),
                invalid_point_value,
                dtype=obj_trans_seg.dtype if obj_trans_seg.size > 0 else float,
            )
            if len_seg > 0:
                is_valid_frame = np.all(obj_trans_seg != invalid_point_value, axis=1)
                valid_indices_in_seg = np.where(is_valid_frame)[0]

                # Find contiguous blocks of valid frames
                if len(valid_indices_in_seg) > 0:
                    block_starts = valid_indices_in_seg[
                        np.concatenate(([True], np.diff(valid_indices_in_seg) > 1))
                    ]
                    block_ends = (
                        valid_indices_in_seg[
                            np.concatenate((np.diff(valid_indices_in_seg) > 1, [True]))
                        ]
                        + 1
                    )

                    for bl_start, bl_end in zip(block_starts, block_ends):
                        block_data = obj_trans_seg[bl_start:bl_end]
                        len_block = block_data.shape[0]

                        if len_block >= min_len_for_smooth:
                            block_data = smooth_translations(
                                block_data,
                                cutoff=cutoff,
                                fs=old_fps,
                                order=filter_order,
                            )

                        upsampled_block = interpolate_translations(
                            block_data, old_fps, new_fps
                        )

                        # Place into the correct upsampled slot
                        ups_bl_start = int(round(bl_start * new_fps / old_fps))
                        ups_bl_end = ups_bl_start + len(upsampled_block)
                        # Ensure slice indices are within bounds of upsampled_obj_trans
                        ups_bl_end = min(ups_bl_end, current_segment_upsampled_len)
                        ups_bl_start = min(
                            ups_bl_start, ups_bl_end
                        )  # ensure start <= end

                        if (
                            upsampled_block.shape[0] > 0
                            and (ups_bl_end - ups_bl_start) > 0
                        ):  # Only assign if there's data and space
                            # Adjust length of upsampled_block if it exceeds available space due to rounding/edge cases
                            upsampled_obj_trans[ups_bl_start:ups_bl_end] = (
                                upsampled_block[: (ups_bl_end - ups_bl_start)]
                            )
            all_processed_segments["object_translations"].append(upsampled_obj_trans)

        if "object_orientations_quat_xyzw" in data_dict:
            obj_rot_seg_quat = data_dict["object_orientations_quat_xyzw"][
                start_idx:end_idx
            ]
            upsampled_obj_rot_quat = np.tile(
                identity_quaternion, (current_segment_upsampled_len, 1)
            ).astype(obj_rot_seg_quat.dtype if obj_rot_seg_quat.size > 0 else float)

            if (
                len_seg > 0 and "object_translations" in data_dict
            ):  # Use object_translations validity
                obj_trans_seg_for_validity = data_dict["object_translations"][
                    start_idx:end_idx
                ]
                is_valid_frame_rot = np.all(
                    obj_trans_seg_for_validity != invalid_point_value, axis=1
                )
                valid_indices_in_seg_rot = np.where(is_valid_frame_rot)[0]

                if len(valid_indices_in_seg_rot) > 0:
                    block_starts_rot = valid_indices_in_seg_rot[
                        np.concatenate(([True], np.diff(valid_indices_in_seg_rot) > 1))
                    ]
                    block_ends_rot = (
                        valid_indices_in_seg_rot[
                            np.concatenate(
                                (np.diff(valid_indices_in_seg_rot) > 1, [True])
                            )
                        ]
                        + 1
                    )

                    for bl_start, bl_end in zip(block_starts_rot, block_ends_rot):
                        block_data_quat = obj_rot_seg_quat[bl_start:bl_end]
                        len_block = block_data_quat.shape[0]
                        if len_block == 0:
                            continue

                        # Check for zero-norm quaternions and replace them
                        norms = np.linalg.norm(block_data_quat, axis=1)
                        zero_norm_mask = norms < 1e-8
                        if np.any(zero_norm_mask):
                            # Ensure we are modifying a copy if necessary, or that block_data_quat can be modified in-place.
                            # Since block_data_quat is a slice, we might need to copy if it's from an array we shouldn't alter directly before this stage.
                            # However, for this specific transformation, it's being converted to rotvecs, so directly modifying this temporary slice is acceptable.
                            block_data_quat_corrected = np.copy(
                                block_data_quat
                            )  # Make a copy to avoid modifying the original slice if it's a view that might be reused elsewhere
                            block_data_quat_corrected[zero_norm_mask] = (
                                identity_quaternion
                            )
                            block_data_rotvec = R.from_quat(
                                block_data_quat_corrected
                            ).as_rotvec()
                            raise ValueError(
                                f"Warning: Found {np.sum(zero_norm_mask)} zero-norm quaternion(s) in an object rotation data block. Replacing with identity quaternion(s)."
                            )
                        else:
                            block_data_rotvec = R.from_quat(block_data_quat).as_rotvec()

                        if len_block >= min_len_for_smooth:
                            block_data_rotvec = smooth_rotations(
                                block_data_rotvec,
                                cutoff=cutoff,
                                fs=old_fps,
                                order=filter_order,
                            )

                        upsampled_block_rotvec = interpolate_rotations(
                            block_data_rotvec, old_fps, new_fps
                        )
                        upsampled_block_quat = R.from_rotvec(
                            upsampled_block_rotvec
                        ).as_quat()

                        ups_bl_start = int(round(bl_start * new_fps / old_fps))
                        ups_bl_end = ups_bl_start + len(upsampled_block_quat)
                        ups_bl_end = min(ups_bl_end, current_segment_upsampled_len)
                        ups_bl_start = min(ups_bl_start, ups_bl_end)

                        if (
                            upsampled_block_quat.shape[0] > 0
                            and (ups_bl_end - ups_bl_start) > 0
                        ):
                            upsampled_obj_rot_quat[ups_bl_start:ups_bl_end] = (
                                upsampled_block_quat[: (ups_bl_end - ups_bl_start)]
                            )
            all_processed_segments["object_orientations_quat_xyzw"].append(
                upsampled_obj_rot_quat
            )

        if use_rubiks_cube_data:
            # Process object_rotation_angles
            if "object_rotation_angles" in data_dict:
                seg_data_angles = data_dict["object_rotation_angles"][start_idx:end_idx]
                if seg_data_angles.size > 0:
                    seg_data_angles_2d = (
                        seg_data_angles.reshape(-1, 1)
                        if seg_data_angles.ndim == 1
                        else seg_data_angles
                    )

                    upsampled_seg_angles_2d = interpolate_translations(
                        seg_data_angles_2d, old_fps, new_fps
                    )
                    upsampled_seg_angles = upsampled_seg_angles_2d.flatten()
                    all_processed_segments["object_rotation_angles"].append(
                        upsampled_seg_angles
                    )
                elif (
                    current_segment_upsampled_len > 0
                ):  # Segment in original was empty, fill with NaN
                    all_processed_segments["object_rotation_angles"].append(
                        np.full(current_segment_upsampled_len, np.nan, dtype=float)
                    )
                else:  # Segment empty and no upsampled frames expected for this segment
                    all_processed_segments["object_rotation_angles"].append(
                        np.array([], dtype=float)
                    )

            # Process object_face_designators
            if "object_face_designators" in data_dict:
                seg_data_faces = data_dict["object_face_designators"][start_idx:end_idx]
                if seg_data_faces.size > 0:
                    upsampled_seg_faces = interpolate_strings(
                        seg_data_faces, old_fps, new_fps, current_segment_upsampled_len
                    )
                    all_processed_segments["object_face_designators"].append(
                        upsampled_seg_faces
                    )
                elif (
                    current_segment_upsampled_len > 0
                ):  # Segment in original was empty, fill with empty strings
                    all_processed_segments["object_face_designators"].append(
                        np.full(current_segment_upsampled_len, "", dtype=object)
                    )
                else:  # Segment empty and no upsampled frames expected
                    all_processed_segments["object_face_designators"].append(
                        np.array([], dtype=object)
                    )

        # Determine actual length of this upsampled segment from primary data type
        # (This ensures consistency if num_upsampled_frames_seg calculation had minor diffs from interpolation results)
        len_this_upsampled_segment = 0
        current_key_order_for_len = [  # Define this list based on what's available
            "hand_translations",
            "hand_orientations_axis_angle",
            "hand_poses",
            "object_translations",
            "object_orientations_quat_xyzw",
            "hand_shapes",
        ]
        if use_rubiks_cube_data:
            current_key_order_for_len.extend(
                ["object_rotation_angles", "object_face_designators"]
            )

        for key_check in current_key_order_for_len:
            if (
                key_check in all_processed_segments  # Check if key exists in dict
                and all_processed_segments[key_check]
                and len(all_processed_segments[key_check][-1]) > 0
            ):
                len_this_upsampled_segment = len(all_processed_segments[key_check][-1])
                break
        if (
            len_this_upsampled_segment == 0 and num_upsampled_frames_seg > 0
        ):  # Fallback if all processed lists empty for this seg but frames were expected
            len_this_upsampled_segment = num_upsampled_frames_seg

        cumulative_upsampled_frames += len_this_upsampled_segment
        if seg_idx < len(iter_boundaries) - 1:  # If not the last segment
            new_segment_start_frames.append(cumulative_upsampled_frames)

    # Concatenate all processed segments for each key
    for key in all_processed_segments:
        if all_processed_segments[key]:
            # Ensure all parts for one key have same dimensions before concat (should be handled by upsampling)
            concatenated_data = np.concatenate(all_processed_segments[key], axis=0)
            processed_data[key] = concatenated_data
        elif (
            key in data_dict
        ):  # Key was in input, but resulted in no data (e.g. original_total_frames=0 or all segments empty)
            processed_data[key] = np.empty(
                (0, data_dict[key].shape[1]), dtype=data_dict[key].dtype
            )
        # else: key was not in input and not processed, so not added to processed_data unless handled in initial empty case.

    processed_data["metadata"]["index_init_frame"] = (
        new_segment_start_frames if new_segment_start_frames else [0]
    )
    processed_data["metadata"]["fps"] = new_fps

    # Check length
    print("\nCheck smoothed and upsampled data")
    expected_length = -1
    # Use keys_to_process which is consistently defined
    for key_check in keys_to_process:
        if (
            key_check in processed_data
            and hasattr(processed_data[key_check], "shape")
            and processed_data[key_check].shape[0] > 0
        ):
            expected_length = processed_data[key_check].shape[0]
            break

    if expected_length == -1:  # If all data arrays are empty
        # Check if cumulative_upsampled_frames provides a length (e.g. from segments that became empty)
        if cumulative_upsampled_frames > 0:
            expected_length = cumulative_upsampled_frames
        else:  # Truly no frames processed or expected
            expected_length = 0

    for key in processed_data:
        if key == "metadata":
            for k, v in processed_data["metadata"].items():
                print(f"data['metadata']['{k}']: {v}")
        else:
            print(f"data['{key}'] shape: {processed_data[key].shape}")
            # Assert only for keys we expect to be processed and conform to sequence length
            if key in keys_to_process:
                assert processed_data[key].shape[0] == expected_length, (
                    f"Length mismatch for {key}: actual {processed_data[key].shape[0]} vs expected {expected_length}"
                )

    return processed_data


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mocap_name", type=str, required=True)
    parser.add_argument("--hand_param_folder", type=str, required=True)
    parser.add_argument("--object_param_folder", type=str, required=True)
    parser.add_argument("--new_fps", default=60, type=int)
    parser.add_argument("--save_original_data", default=False, type=bool)
    parser.add_argument("--save_path", default=None, type=str, required=False)
    args = parser.parse_args()

    # Load the data
    data_dict = get_motion_object_data(
        args.mocap_name, args.hand_param_folder, args.object_param_folder
    )
    data_dict = change_mano_translation_to_root_translation(data_dict)

    # Save the data as npz
    if args.save_original_data:
        np.savez(args.save_path, **data_dict)
    else:
        smoothed_data_dict = smooth_data(data_dict, new_fps=args.new_fps)
        np.savez(args.save_path, **smoothed_data_dict)


if __name__ == "__main__":
    __main__()
