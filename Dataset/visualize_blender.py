import argparse
import os

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from ObjectReconstruction.rubikscube import RubiksCube222
from typing import Dict

MANO_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    os.path.pardir,
    "HandReconstruction",
    "Data",
    "HumanModels",
)

from videoio import VideoWriter
from imageio import imwrite

import bpy
from blendify import scene
from blendify.colors import UniformColors, FacesUV, FileTextureColors
from blendify.materials import PrincipledBSDFMaterial
from blendify.utils.image import blend_with_background
from blendify.utils.smpl_wrapper import SMPLWrapper
import blendify.renderables.collection

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



def visualize_rubiks_cube_animation_blender(
    animation_data: dict,
    animation_fps: int = 120,
    cubelet_size: float = 1.0,
    gap: float = 0.02,
    cube_name: str = "RubiksCube222",
):
    """
    Visualizes a Rubik's Cube animation based on per-frame global transformations and face rotations.

    Parameters:
    - animation_data (dict): A dictionary containing the core animation data.
        Expected keys:
        - "translations" (np.ndarray, shape=(N, 3)): Global translations for N frames.
        - "orientations" (np.ndarray, shape=(N, 3, 3)): Global orientations (rotation matrices) for N frames.
        - "face_designators" (np.ndarray, shape=(N,), dtype=object): Face designator ("U", "D", etc., or None) for each frame.
        - "rotation_angles" (np.ndarray, shape=(N,), dtype=float): Incremental rotation angle in **radians** for the
                                                                  designated face for each frame.
    - animation_fps (int): Frames per second for the animation.
    - cubelet_size (float): Side length of a single small cubelet.
    - gap (float): Visual gap between adjacent cubelets.
    - cube_name (str): Name for the Rerun application.
    """
    # Unpack parameters from the animation_data dictionary
    all_global_translations = animation_data["translations"]
    all_global_orientations_mat = animation_data["orientations"]
    face_designators_per_frame = animation_data["face_designators"]
    angles_per_frame = animation_data["rotation_angles"]

    # Other parameters are now direct arguments
    # animation_fps, cubelet_size, gap, rerun_app_name, save_rerun_recording are used directly.
    rubiks_cube = RubiksCube222(cubelet_size=cubelet_size, gap=gap, name=cube_name)

    total_animation_frames = all_global_translations.shape[0]

    if total_animation_frames == 0:
        print(
            "Warning: No global transformation frames provided. Nothing to visualize."
        )
        return

    # --- Log the initial rest pose at time 0 ---
    ret = rubiks_cube.get_rubik_mesh(
        overall_translation=all_global_translations[0],
        overall_orientation_mat=all_global_orientations_mat[0],
    )
    i_frame = yield ret
    if i_frame is None:
        return
    # -----------------------------------------

    # Validate shapes of provided arrays
    if all_global_orientations_mat.shape[0] != total_animation_frames:
        raise ValueError(
            f"Shape mismatch: `orientations` has {all_global_orientations_mat.shape[0]} frames, "
            f"expected {total_animation_frames} (from `translations`)."
        )
    if face_designators_per_frame.shape[0] != total_animation_frames:
        raise ValueError(
            f"Shape mismatch: `face_designators` has {face_designators_per_frame.shape[0]} frames, "
            f"expected {total_animation_frames}."
        )
    if angles_per_frame.shape[0] != total_animation_frames:
        raise ValueError(
            f"Shape mismatch: `rotation_angles` has {angles_per_frame.shape[0]} frames, "
            f"expected {total_animation_frames}."
        )

    # Main animation loop
    # for i_frame in range(total_animation_frames):
    while True:
        current_time_sec = (i_frame) / float(animation_fps)
        # rr.set_time_seconds("timestamp", current_time_sec) # Changed back

        # Apply incremental rotation for the current frame if specified
        face_designator = face_designators_per_frame[i_frame]
        incremental_angle_rad = angles_per_frame[i_frame]

        if face_designator and incremental_angle_rad != 0.0:
            # print(f"Frame {i_frame} (t={current_time_sec:.3f}s): Rotating {face_designator} by {incremental_angle_rad:.4f} rad")
            rubiks_cube.rotate_rubiks_face(face_designator, incremental_angle_rad)

        current_translation = all_global_translations[i_frame]
        current_orientation = all_global_orientations_mat[i_frame]

        ret = rubiks_cube.get_rubik_mesh(
            overall_translation=current_translation,
            overall_orientation_mat=current_orientation,
        )
        
        i_frame = yield ret
        if i_frame is None or i_frame < 0 or i_frame >= total_animation_frames:
            break

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
    
    parser.add_argument("-p", "--path", type=str, default="./05_smpl_movement.mp4",
                        help="Path to the resulting video")
    parser.add_argument("-o", "--output-blend", type=str, default=None,
                        help="Path to the resulting blend file")

    # Rendering parameters
    parser.add_argument("-n", "--n-samples", default=256, type=int,
                        help="Number of paths to trace for each pixel in the render (default: 256)")
    parser.add_argument("-res", "--resolution", default=(1024, 1024), nargs=2, type=int,
                        help="Rendering resolution, (default: (1280, 720))")
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU for rendering (by default GPU is used)")
    parser.add_argument("-sk", "--skip_download", action="store_true",
                        help="Skip asset downloads")
    
    
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

    # Print data
    for key, value in data.items():
        if key == "metadata":
            continue
        print(f"data['{key}'].shape: {value.shape}")

    # Read data
    optimized_translations = data["hand_translations"]
    optimized_orientations = data["hand_orientations_axis_angle"]
    optimized_poses = data["hand_poses"]
    optimized_shapes = data["hand_shapes"]
    num_frames_hand = optimized_translations.shape[0]

    use_rubiks_cube_data = False
    if "RubiksCube" in mocap_session_name_base:
        use_rubiks_cube_data = True

    object_translations = data["object_translations"]
    object_orientations_quat_xyzw = data["object_orientations_quat_xyzw"]
    if use_rubiks_cube_data:
        object_face_designators = data["object_face_designators"]
        object_rotation_angles = data["object_rotation_angles"]
    num_frames_object = object_translations.shape[0]

    assert num_frames_hand == num_frames_object, (
        f"Number of frames in hand data ({num_frames_hand}) does not match number of frames in object data ({num_frames_object})."
    )

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
    for segment_idx, segment_bounds in enumerate(index_init_frame):
        if segment_idx == len(index_init_frame) - 1:
            index_init_frame_segments.append([segment_bounds, num_frames_hand])
        else:
            index_init_frame_segments.append(
                [segment_bounds, index_init_frame[segment_idx + 1]]
            )
            
    ## setup blender
    
    # Set custom parameters to improve quality of rendering
    scene.attach_blend(os.path.join(os.getcwd(), "./Dataset/light_box.blend"))
    bpy.context.scene.cycles.max_bounces = 5 # 30
    bpy.context.scene.cycles.transmission_bounces = 4 # 20
    bpy.context.scene.cycles.transparent_max_bounces = 5 # 15
    bpy.context.scene.cycles.diffuse_bounces = 4 # 10
    bpy.context.scene.cycles.denoising_store_passes = True
    # bpy.context.scene.view_layers[0].cycles.use_denoising = True
    
    # scene.set_perspective_camera(args.resolution, fov_x=0.7, rotation=(0.82, 0.42, 0.18, 0.34), translation=(5, -5, 5))
    # camera = scene.set_perspective_camera(args.resolution, fov_y=np.deg2rad(75))    
    scene.set_perspective_camera(
        args.resolution, fov_x=np.deg2rad(20.8),
        translation=(0.1, -0.56, 0.43),
        rotation=(0.889, 0.458, 0, 0),
    )
    
    # scene.lights.add_point(strength=1000, translation=(4, -2, 4))
    
    # Define the materials
    # Material and Colors for SMPL mesh
    smpl_material = PrincipledBSDFMaterial()
    # smpl_colors = UniformColors((0.3, 0.3, 0.3))    
    smpl_colors = UniformColors((51/255, 204/255, 204/255))
    
    # Set the lights; one main sunlight and a secondary light without visible shadows to make the scene overall brighter
    # sunlight = scene.lights.add_sun(
    #     strength=2.3, rotation_mode="euleryz", rotation=(-45, -90)
    # )
    # sunlight2 = scene.lights.add_sun(
    #     strength=3, rotation_mode="euleryz", rotation=(-45, 165)
    # )
    # sunlight2.cast_shadows = False
    
    scene.lights.add_point(rotation=(0.571, 0.169, 0.272, 0.756), translation=(21.0, 0.0, 7.0), strength=12000)
    scene.lights.add_point(rotation=(0.571, 0.169, 0.272, 0.756), translation=(0.0, -21, 7.0), strength=10000)
    
    # Add the SMPL mesh, set the pose to zero for the first frame, just to initialize
    smpl_vertices = hand_vertices[0]
    smpl_faces = smplx_model.faces
    smpl_mesh = scene.renderables.add_mesh(smpl_vertices, smpl_faces, smpl_material, smpl_colors)
    smpl_mesh.set_smooth()  # Force the surface of model to look smooth
    
    
    # initialize geometry
    if use_rubiks_cube_data:
        rubiks_cube_mesh_gen = visualize_rubiks_cube_animation_blender(
            animation_data={
                    "translations": object_translations,
                    "orientations": R.from_quat(
                        object_orientations_quat_xyzw
                    ).as_matrix(),
                    "face_designators": object_face_designators,
                    "rotation_angles": object_rotation_angles,
                },
                animation_fps=fps,
                cubelet_size=object_size[0] / 2,
                gap=0.0,
                cube_name=f"{mocap_session_name_base}_object",
            )
        
        init_rubiks_cube = rubiks_cube_mesh_gen.send(None)
        
        rubiks_cube_meshes:Dict[str, blendify.renderables.collection.Mesh] = {}
        for entity_path, mesh_args in init_rubiks_cube.items():
            # print(mesh_args['vertex_colors'])
            rubiks_cube_meshes[entity_path] = scene.renderables.add_mesh(
                vertices=mesh_args['vertex_positions'],
                faces=mesh_args['triangle_indices'],
                material=PrincipledBSDFMaterial(),
                colors=UniformColors(mesh_args['vertex_colors'][0]/255),
            )
            rubiks_cube_meshes[entity_path].set_smooth(False)
        
    else:        
        # add cube 
        # Create material
        cube_material = PrincipledBSDFMaterial()
        # Create color
        cube_color = UniformColors((1.0, 0.0, 0.0))
        cube = scene.renderables.add_cube_mesh(1.0, cube_material, cube_color, scale=object_size)
        cube.set_smooth(False)

    # Visualize each segment
    def render():
        first_seen_obj = True
        for segment_idx, segment_bounds in enumerate(index_init_frame_segments):
            segment_data_start_idx = segment_bounds[0]
            segment_data_end_idx = segment_bounds[1]

            print(
                f"\nVisualizing Segment {segment_idx}: frames {segment_data_start_idx} to {segment_data_end_idx} (inclusive global indices). "
            )

            # Visualize hand for this segment (subsequent frames as partial updates)
            for frame_idx in tqdm(
                range(segment_data_start_idx, segment_data_end_idx),
                desc=f"Hand Segment {segment_idx} Updates",
            ):
                if frame_idx >= num_frames_hand:
                    break
                
                if first_seen_obj and np.allclose(
                    object_translations[frame_idx], np.array(invalid_point_value)
                ):
                    continue
                
                first_seen_obj = False

                current_hand_verts = hand_vertices[frame_idx]
                # current_hand_normals = compute_vertex_normals(
                #     current_hand_verts, smplx_model.faces
                # )
                smpl_mesh.update_vertices(current_hand_verts)
                
                print(
                    f"frame: {frame_idx}/{segment_data_end_idx}"
                )
                
                # Rubiks cube data is only available for the first 8400 frames
                if use_rubiks_cube_data and frame_idx >= 8400:
                    break

                if frame_idx >= num_frames_object:
                    break
         
                if use_rubiks_cube_data:                    
                    rubiks_cube = rubiks_cube_mesh_gen.send(frame_idx)
                    for entity_path, mesh_args in rubiks_cube.items():
                        rubiks_cube_meshes[entity_path].update_vertices(mesh_args['vertex_positions'])
                        
                else:
                    cube.set_position(rotation_mode="quaternionXYZW", 
                                      rotation=object_orientations_quat_xyzw[frame_idx], 
                                      translation=object_translations[frame_idx])
                    
                    
                
                # scene.render(filepath="cube.png", use_gpu=not args.cpu, samples=args.n_samples)
                # exit(0)
                
                img = scene.render(use_gpu=not args.cpu, samples=args.n_samples)
                # Frames have transparent background; perform an alpha blending with white background instead
                img_white_bkg = blend_with_background(img, (1.0, 1.0, 1.0))
                # Add the frame to the video
                
                yield frame_idx, img_white_bkg

    
    renderer = render()
    if os.path.isdir(args.path):
        for frame_idx, img in renderer:
            img_fn = os.path.join(args.path, f'img_{frame_idx:08d}.png')
            imwrite(img_fn, img)
            
    else:
        with VideoWriter(args.path, resolution=args.resolution, fps=30) as vw:
            for frame_idx, img in renderer:
                vw.write(img)

if __name__ == "__main__":
    main()
