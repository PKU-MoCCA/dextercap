import os

# The unique string represents this mocap session
mocap_session_name = "Cuboid_00"

# Hand related configuration
hand = {
    # 指关节顺序：食指、中指、小指、无名指、拇指, 每个手指有三个关节
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
    # 关节数量, 共有 16 个关节
    "total_joint_num": 16,
    # 粘贴的 patch 数量, 根关节对应两个 patch, 所以是 17
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
    # 手背手心至少需要多少marker才会进行根节点位置和旋转的优化
    "min_marker_for_calibration": 1,
    # calibration 的帧数
    "num_frame_for_calibration": 250,
    # calibration 和 optimization 的迭代次数
    "num_iterations": 400,
    # 哪些帧是手重新出现的帧, 代表这一帧和上一帧是两个动作序列, 如果所有序列是连续的, 则 index_init_frame = [0], 如果第十帧手从相机外面进来了, 则 index_init_frame = [0, 10]
    "index_init_frame": [0, 1333],
    # 手重新出现的帧的迭代次数
    "num_iterations_init_frame": 1000,
}
