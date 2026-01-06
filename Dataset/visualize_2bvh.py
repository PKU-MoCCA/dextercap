import argparse
import os

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from pymotionlib.MotionData import MotionData
import pymotionlib.BVHLoader as BVH


MANO_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    os.path.pardir,
    "HandReconstruction",
    "Data",
    "HumanModels",
)

JOINT_NAMES = [
    'wrist',
    'index1', 'index2', 'index3',
    'middle1', 'middle2', 'middle3',
    'pinky1', 'pinky2', 'pinky3',
    'ring1', 'ring2', 'ring3',
    'thumb1', 'thumb2', 'thumb3',
    'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip', 'thumb_tip',
]
# Define MANO hand skeleton connections (parent-child joint indices)
# These are typical for MANO, adjust if your model differs
MANO_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),        # Index
    (0, 4), (4, 5), (5, 6),        # Middle
    (0, 7), (7, 8), (8, 9),        # Pinky
    (0, 10), (10, 11), (11, 12),   # Ring
    (0, 13), (13, 14), (14, 15),   # Thumb
]

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
        "--mano_model_path",
        type=str,
        default=MANO_MODEL_PATH,
    )
    parser.add_argument(
        "--bvh_path",
        type=str,
        default="",
        help="Path to the output BVH file.",
    )

    args = parser.parse_args()

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

    use_rubiks_cube_data = False
    if "RubiksCube" in mocap_session_name_base:
        use_rubiks_cube_data = True

    # Read data
    optimized_translations = (
        data["hand_translations"]
        if not use_rubiks_cube_data
        else data["hand_translations"][:8400]
    )
    optimized_orientations = (
        data["hand_orientations_axis_angle"]
        if not use_rubiks_cube_data
        else data["hand_orientations_axis_angle"][:8400]
    )
    optimized_poses = (
        data["hand_poses"] if not use_rubiks_cube_data else data["hand_poses"][:8400]
    )
    optimized_shapes = (
        data["hand_shapes"] if not use_rubiks_cube_data else data["hand_shapes"][:8400]
    )
    num_frames_hand = optimized_translations.shape[0]

    object_translations = (
        data["object_translations"]
        if not use_rubiks_cube_data
        else data["object_translations"][:8400]
    )
    object_orientations_quat_xyzw = (
        data["object_orientations_quat_xyzw"]
        if not use_rubiks_cube_data
        else data["object_orientations_quat_xyzw"][:8400]
    )
    if use_rubiks_cube_data:
        object_face_designators = data["object_face_designators"][:8400]
        object_rotation_angles = data["object_rotation_angles"][:8400]
    num_frames_object = object_translations.shape[0]

    assert num_frames_hand == num_frames_object, (
        f"Number of frames in hand data ({num_frames_hand}) does not match number of frames in object data ({num_frames_object})."
    )

    # Print data
    for key, value in data.items():
        if key == "metadata":
            continue
        print(f"data['{key}'].shape: {value.shape}")

    # Initialize MANO model and calculate hand vertices and joints
    smplx_model = smplx.create(
        model_path=args.mano_model_path,
        model_type="mano",
        flat_hand_mean=True,
        is_rhand=False,
        use_pca=False,
        batch_size=num_frames_hand,
    )
    # we need the first frame to be the reference pose
    optimized_orientations[0] = np.zeros(3, dtype=np.float32)
    optimized_translations[0] = np.zeros(3, dtype=np.float32)
    optimized_poses[0] = np.zeros(45, dtype=np.float32)
    
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
    print(smplx_model)
    print(smplx_model.name())
    print(smplx_model.parents)
    print(smplx_model.posedirs)
    print(optimized_shapes)
    
    smplx_output = smplx_model(**hand_parms)
    hand_vertices = smplx_output.vertices.detach().cpu().numpy()
    hand_joints = smplx_output.joints.detach().cpu().numpy()
    parent_indices = smplx_model.parents.detach().cpu().numpy()
    
    for frame_idx in range(num_frames_hand):
        R_hand = R.from_rotvec(optimized_orientations[frame_idx])
        hand_vertices[frame_idx] = (
            R_hand.apply(hand_vertices[frame_idx])
            + optimized_translations[frame_idx]
        )
        hand_joints[frame_idx] = (
            R_hand.apply(hand_joints[frame_idx]) + optimized_translations[frame_idx]
        )
        
    new_face_index = [
        121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120, 108, 79, 78,
    ]
    more_face = []  # To make MANO as a whole watertight mesh
    for i in range(2, len(new_face_index)):
        more_face.append([121, new_face_index[i - 1], new_face_index[i]])
    hand_faces = np.concatenate([smplx_model.faces, more_face], axis=0)
    
    
    tips_parents = [3, 6, 9, 12, 15]  # Indices of the tips in the MANO model
    bvh_data = MotionData()
    bvh_data._num_joints = 16 + 5 # 16 MANO joints + 5 tips
    bvh_data._skeleton_joints = JOINT_NAMES
    bvh_data._skeleton_joint_parents = parent_indices.tolist() + tips_parents  # Add tips as children of the last joint
    bvh_data._skeleton_joint_offsets = hand_joints[0] - hand_joints[0, parent_indices]
    bvh_data._skeleton_joint_offsets[0] = np.zeros(3)  # Root joint offset is zero
    bvh_data._skeleton_joint_offsets = np.concatenate(
        [bvh_data._skeleton_joint_offsets, bvh_data._skeleton_joint_offsets[tips_parents]], axis=0
    )  # Add zero offsets for tips
    bvh_data._end_sites = [16, 17, 18, 19, 20]  # Indices of the tips in the BVH data
    
    bvh_data._num_frames = num_frames_hand
    bvh_data._fps = fps
    bvh_data._joint_rotation = np.zeros((bvh_data._num_frames, bvh_data._num_joints, 4), dtype=np.float32)
    bvh_data._joint_rotation[...,-1] = 1.0
    bvh_data._joint_rotation[:,0] = R.from_rotvec(optimized_orientations).as_quat()
    bvh_data._joint_rotation[:,1:16] = R.from_rotvec(optimized_poses.reshape(-1, 3)).as_quat().reshape(num_frames_hand, 15, 4)
    bvh_data._joint_translation = np.zeros((bvh_data._num_frames, bvh_data._num_joints, 3), dtype=np.float32)
    bvh_data._joint_translation[:,0] = hand_joints[:, 0]
    # optimized_translations - hand_joints[:1,0] # hand_joints[:, 0] - hand_joints[:1,0]

    bvh_file_path = args.bvh_path 
    if not bvh_file_path:
        bvh_dir = os.path.join(os.path.dirname(args.data_path), 'bvh')
        os.makedirs(bvh_dir, exist_ok=True)
        bvh_file_path = os.path.join(bvh_dir, os.path.basename(args.data_path))
        bvh_file_path = bvh_file_path.replace(".npz", ".bvh")
        assert bvh_file_path != args.data_path
            
    if not bvh_file_path.endswith(".bvh"):
        bvh_file_path += ".bvh"

    # Save BVH data
    BVH.save(bvh_data, bvh_file_path[:-4] + "_hand.bvh")
    
    # save object bvh
    object_bvh_data = MotionData()
    object_bvh_data._num_joints = 2  # Two joints for the object
    object_bvh_data._skeleton_joints = ["object", "object_end"]
    object_bvh_data._skeleton_joint_parents = [-1, 0]  # No parent for the root joint
    object_bvh_data._skeleton_joint_offsets = np.zeros((2, 3), dtype=np.float32)
    object_bvh_data._skeleton_joint_offsets[1] = [0, 0, max(object_size)]  # End joint offset is at the top of the object
    object_bvh_data._end_sites = [1]
    
    object_bvh_data._num_frames = num_frames_object
    object_bvh_data._fps = fps
    object_bvh_data._joint_rotation = np.zeros((object_bvh_data._num_frames, object_bvh_data._num_joints, 4), dtype=np.float32)
    object_bvh_data._joint_rotation[..., -1] = 1.0
    object_bvh_data._joint_rotation[:, 0] = R.from_quat(object_orientations_quat_xyzw).as_quat()
    object_bvh_data._joint_translation = np.zeros((object_bvh_data._num_frames, object_bvh_data._num_joints, 3), dtype=np.float32)
    object_bvh_data._joint_translation[:, 0] = object_translations
    
    BVH.save(object_bvh_data, bvh_file_path[:-4] + "_object.bvh")

    print(object_translations[3000])
    print(optimized_translations[3000])
    print(hand_joints[3000])
    

if __name__ == "__main__":
    main()
