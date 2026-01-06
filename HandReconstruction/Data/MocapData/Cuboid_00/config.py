import os

# The unique string represents this mocap session
mocap_session_name = "Cuboid_00"

# Hand related configuration
hand = {
    # Knuckle order: index finger, middle finger, little finger, ring finger, thumb. Each finger has three joints.
    "dof": [
        3,
        1,
        1,
        3,
        1,
        1,
        3,
        1,
        1,
        3,
        1,
        1,
        3,
        3,
        1,
    ],
    # Number of joints: a total of 16 joints
    "total_joint_num": 16,
    # Number of pasted patches: the root joint corresponds to two patches, so it is 17.
    "total_segment_num": 17,
    "MANO_shape": [
        0.25624018907546997,
        -0.46635702252388,
        -1.6472688913345337,
        0.17522339522838593,
        -0.09414590150117874,
        0.9134469628334045,
        0.21890263259410858,
        1.025447964668274,
        0.03658989071846008,
        -0.29835939407348633,
    ],
}

# Data related configuration
data = {
    "smpl_model_path": os.path.join(os.path.dirname(__file__), "Data", "HumanModels"),
    "body_part_index_path": os.path.join(
        os.path.dirname(__file__),
        "Data",
        "MocapData",
        mocap_session_name,
        "point_segment_idx-2025.04.21.npy",
    ),
    "hand_mocap_point_data_path": os.path.join(
        os.path.dirname(__file__),
        "Data",
        "MocapData",
        mocap_session_name,
        "mocap_point-2025.04.23.npy",
    ),
    "invalid_point_value": [-1000, -1000, -1000],
    "cutoff_end": -1,
    "cutoff_start": 0,
    "cutoff_step": 1,
    "fps": 20,
    "save_one_frame_curve_every_n_frames": 100,
    "save_rerun_and_MANO_every_n_frames": 500,
}

# Optimization related configuration
optimize = {
    "test_mode": False,
    "device": "cpu",
    "use_shape_fitting": True,
    "learning_rate": 0.002,
    "end_effector_index": [3, 6, 9, 12, 15],
    "weight_for_end_effector": 3,
    "regularization_weight": 1000,
    # At least how many markers are needed on the back and palm of the hand to optimize the root node position and rotation?
    "min_marker_for_calibration": 1,
    # Number of frames for calibration
    "num_frame_for_calibration": 250,
    # Number of iterations for calibration and optimization
    "num_iterations": 400,
    # Which frames are the frames where the hand reappears, representing that this frame and the previous frame are two action sequences. If all sequences are continuous, then index_init_frame = [0], if the 10th frame of the hand comes into the camera from outside, then index_init_frame = [0, 10]
    "index_init_frame": [0, 1333],
    # Number of iterations for the frames where the hand reappears
    "num_iterations_init_frame": 1000,
}
