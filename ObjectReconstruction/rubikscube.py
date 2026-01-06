import numpy as np
import rerun as rr
import transforms3d as t3d

from HandReconstruction.Utility.utils_visualize import init_rerun

# --- Constants ---

COLORS = {
    "yellow": (255, 255, 0),
    "white": (255, 255, 255),
    "orange": (255, 120, 0),
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 128, 0),
    "hidden": (255, 255, 255),  # Internal faces are white
}

# (axis_index, direction, color_name) for solved state
# axis_index: 0 for X, 1 for Y, 2 for Z
# direction: +1 or -1
FACE_COLORS_SOLVED = {
    (0, 1): "red",  # +X (Right)
    (0, -1): "orange",  # -X (Left)
    (1, 1): "yellow",  # +Y (Up)
    (1, -1): "white",  # -Y (Down)
    (2, 1): "blue",  # +Z (Front)
    (2, -1): "green",  # -Z (Back)
}

CUBELET_SIZE = 1.0
HALF_CUBELET_SIZE = CUBELET_SIZE / 2.0

# Vertices for a single face centered at origin, on XY plane, normal along +Z
BASE_FACE_VERTICES = np.array(
    [
        [-HALF_CUBELET_SIZE, -HALF_CUBELET_SIZE, 0],
        [HALF_CUBELET_SIZE, -HALF_CUBELET_SIZE, 0],
        [HALF_CUBELET_SIZE, HALF_CUBELET_SIZE, 0],
        [-HALF_CUBELET_SIZE, HALF_CUBELET_SIZE, 0],
    ],
    dtype=np.float32,
)

BASE_FACE_TRIANGLES = np.array(
    [
        [0, 1, 2],
        [0, 2, 3],
    ],
    dtype=np.uint32,
)

# Transformations to get from base face (on XY plane, normal +Z) to each of the 6 cubelet faces
# Each is a rotation matrix. Keys like "PX" means "Positive X direction"
FACE_ORIENTATIONS_IN_CUBELET = {
    "PX": t3d.axangles.axangle2mat(
        [0, 1, 0], np.pi / 2
    ),  # Rotated around Y by 90 deg: Z-normal -> X-normal
    "NX": t3d.axangles.axangle2mat(
        [0, 1, 0], -np.pi / 2
    ),  # Rotated around Y by -90 deg: Z-normal -> -X-normal
    "PY": t3d.axangles.axangle2mat(
        [1, 0, 0], -np.pi / 2
    ),  # Rotated around X by -90 deg: Z-normal -> Y-normal
    "NY": t3d.axangles.axangle2mat(
        [1, 0, 0], np.pi / 2
    ),  # Rotated around X by 90 deg: Z-normal -> -Y-normal
    "PZ": np.eye(3),  # No rotation needed for +Z normal
    "NZ": t3d.axangles.axangle2mat(
        [1, 0, 0], np.pi
    ),  # Rotated around X by 180 deg: Z-normal -> -Z-normal
}


class RubiksCube222:
    """
    Represents a 2x2x2 Rubik's Cube for Rerun visualization.
    The cube is composed of 8 cubelets, each with 6 faces.
    Each face is logged as an individual mesh to Rerun.
    """

    def __init__(
        self, cubelet_size: float = 1.0, gap: float = 0.05, name: str = "RubiksCube222"
    ):
        """
        Initializes the Rubik's Cube.

        Parameters:
        - cubelet_size (float): The side length of a single small cubelet.
        - gap (float): The visual gap between adjacent cubelets.
        """
        self.cubelet_size = cubelet_size
        self.half_cubelet_size = cubelet_size / 2.0
        self.gap = gap
        self.actual_cubelet_spacing = self.cubelet_size + self.gap
        self.name = name

        self.cubelets = []  # List of dictionaries, each representing a cubelet
        self._initialize_cubelets()

    def _get_face_color(
        self,
        cubelet_pos_id: tuple[int, int, int],
        face_normal_in_cubelet_frame: tuple[int, int],
    ) -> tuple[int, int, int]:
        """
        Determines the color of a cubelet's face in the solved state.
        Internal faces are colored white.

        Parameters:
        - cubelet_pos_id (tuple[int, int, int]): Identifier for the cubelet's position (-1 or 1 for x, y, z).
                                                 e.g., (-1, -1, -1) is the bottom-left-back cubelet.
        - face_normal_in_cubelet_frame (tuple[int, int]): (axis_index, direction) of the face's normal
                                                          in the cubelet's local coordinate system.
                                                          e.g., (0, 1) means face normal is along +X in cubelet's frame.

        Returns:
        - tuple[int, int, int]: RGB color tuple (0-255).
        """
        for axis_idx in range(3):  # 0:X, 1:Y, 2:Z
            # Check if the face is an outer face of the Rubik's cube assembly
            if (
                cubelet_pos_id[axis_idx] == face_normal_in_cubelet_frame[1]
                and axis_idx == face_normal_in_cubelet_frame[0]
            ):
                # This face is on the "outside" of the Rubik's cube.
                # Its color is determined by its world orientation in the solved state.
                # The key for FACE_COLORS_SOLVED is (axis_idx_world, direction_world)
                world_axis_idx = axis_idx
                world_direction = cubelet_pos_id[axis_idx]
                return COLORS[FACE_COLORS_SOLVED[(world_axis_idx, world_direction)]]

        return COLORS["hidden"]  # Internal face

    def _initialize_cubelets(self):
        """
        Creates the 8 cubelets with their initial positions, orientations, and face data.
        """
        spacing_center = self.actual_cubelet_spacing / 2.0
        cube_idx_counter = 0

        for iz_id in [-1, 1]:  # Back/Front layer
            for iy_id in [-1, 1]:  # Bottom/Top layer
                for ix_id in [-1, 1]:  # Left/Right layer
                    cubelet_pos_id = (
                        ix_id,
                        iy_id,
                        iz_id,
                    )  # Identifier based on position

                    # Initial center position of the cubelet in the Rubik's cube's frame
                    initial_center_pos = np.array(
                        [
                            ix_id * spacing_center,
                            iy_id * spacing_center,
                            iz_id * spacing_center,
                        ]
                    )

                    faces_data = []
                    # Define 6 faces for this cubelet based on their normal directions in cubelet's local frame
                    # (face_orientation_key, (axis_idx_in_cubelet, direction_in_cubelet))
                    face_definitions = [
                        ("PX", (0, 1)),
                        ("NX", (0, -1)),  # +X, -X
                        ("PY", (1, 1)),
                        ("NY", (1, -1)),  # +Y, -Y
                        ("PZ", (2, 1)),
                        ("NZ", (2, -1)),  # +Z, -Z
                    ]

                    for i, (face_key, normal_in_cubelet) in enumerate(face_definitions):
                        color = self._get_face_color(cubelet_pos_id, normal_in_cubelet)

                        # Rotation of the face relative to the cubelet's center
                        rot_mat_face_in_cubelet = FACE_ORIENTATIONS_IN_CUBELET[face_key]

                        # Translation of the face center from the cubelet center
                        # Normal vector in cubelet frame for this face
                        face_normal_vec = np.zeros(3)
                        face_normal_vec[normal_in_cubelet[0]] = normal_in_cubelet[1]
                        trans_vec_face_center = face_normal_vec * self.half_cubelet_size

                        # Homogeneous transform matrix for the face relative to cubelet's center
                        T_face_in_cubelet = np.eye(4)
                        T_face_in_cubelet[:3, :3] = rot_mat_face_in_cubelet
                        T_face_in_cubelet[:3, 3] = trans_vec_face_center

                        faces_data.append(
                            {
                                "entity_id_suffix": f"face_{face_key}",  # Unique part of entity path
                                "color": color,
                                "transform_in_cubelet": T_face_in_cubelet,  # Static transform of face wrt cubelet
                            }
                        )

                    self.cubelets.append(
                        {
                            "id_tuple": cubelet_pos_id,  # e.g. (-1, 1, 1)
                            "id_numeric": cube_idx_counter,
                            "current_position_in_rubiks": initial_center_pos.copy(),
                            "current_orientation_in_rubiks": np.eye(
                                3
                            ),  # Rotation matrix
                            "faces": faces_data,
                        }
                    )
                    cube_idx_counter += 1

    def log_to_rerun(
        self,
        overall_translation: np.ndarray = np.zeros(3),
        overall_orientation_mat: np.ndarray = np.eye(3),
    ):
        """
        Logs the current state of the Rubik's cube to Rerun.
        Each of the 48 faces is logged as an individual `rr.Mesh3D` entity.

        Parameters:
        - overall_translation (np.ndarray, shape=(3,)): Translation to apply to the entire Rubik's cube.
        - overall_orientation_mat (np.ndarray, shape=(3,3)): Rotation matrix to apply to the entire Rubik's cube.
        """

        ret = self.get_rubik_mesh(overall_translation, overall_orientation_mat)
        for entity_path, mesh_args in ret.items():
            rr.log(
                entity_path,
                rr.Mesh3D(
                    vertex_positions=mesh_args["vertex_positions"],
                    triangle_indices=mesh_args["triangle_indices"],
                    vertex_colors=mesh_args["vertex_colors"],
                    vertex_normals=mesh_args["vertex_normals"],
                ),
            )

    def get_rubik_mesh(
        self,
        overall_translation: np.ndarray = np.zeros(3),
        overall_orientation_mat: np.ndarray = np.eye(3),
    ):
        """
        return the current state of the Rubik's cube for logging
        Each of the 48 faces is returned in dict

        Parameters:
        - overall_translation (np.ndarray, shape=(3,)): Translation to apply to the entire Rubik's cube.
        - overall_orientation_mat (np.ndarray, shape=(3,3)): Rotation matrix to apply to the entire Rubik's cube.
        """
        T_rubiks_in_world_overall = np.eye(4)
        T_rubiks_in_world_overall[:3, :3] = overall_orientation_mat
        T_rubiks_in_world_overall[:3, 3] = overall_translation

        base_normal_face_local = np.array(
            [0, 0, 1], dtype=np.float32
        )  # Normal of BASE_FACE_VERTICES (before its own rotation)

        ret = {}

        for cubelet in self.cubelets:
            # Transform of the cubelet in the Rubik's cube's frame (which can be rotated/translated itself)
            T_cubelet_in_rubiks = np.eye(4)
            T_cubelet_in_rubiks[:3, :3] = cubelet["current_orientation_in_rubiks"]
            T_cubelet_in_rubiks[:3, 3] = cubelet["current_position_in_rubiks"]

            # Full transform for this cubelet from its local space to world space
            T_cubelet_in_world = T_rubiks_in_world_overall @ T_cubelet_in_rubiks

            for face_data in cubelet["faces"]:
                entity_path = f"{self.name}/c{cubelet['id_numeric']}/{face_data['entity_id_suffix']}"

                T_face_in_cubelet = face_data["transform_in_cubelet"]

                # Final transform for this specific face in world coordinates
                T_final_face_in_world = T_cubelet_in_world @ T_face_in_cubelet

                # Scale base vertices according to the instance's cubelet_size
                # BASE_FACE_VERTICES is defined for a 1x1 (unit) cubelet face part.
                scaled_base_face_vertices = BASE_FACE_VERTICES * self.cubelet_size

                # Transform scaled base vertices to world space
                base_verts_homog = np.hstack(
                    (
                        scaled_base_face_vertices,
                        np.ones((scaled_base_face_vertices.shape[0], 1)),
                    )
                )
                transformed_verts_homog = (T_final_face_in_world @ base_verts_homog.T).T
                transformed_verts = transformed_verts_homog[:, :3].astype(np.float32)

                # Transform face normal to world space
                # The normal of the base face (before T_face_in_cubelet) is (0,0,1)
                # The rotation part of T_final_face_in_world transforms this normal
                R_final_face = T_final_face_in_world[:3, :3]
                transformed_normal = (R_final_face @ base_normal_face_local).astype(
                    np.float32
                )
                # Rerun typically normalizes normals if needed, but good practice:
                # transformed_normal /= np.linalg.norm(transformed_normal)

                face_vertex_normals = np.tile(
                    transformed_normal, (BASE_FACE_VERTICES.shape[0], 1)
                )

                ret[entity_path] = dict(
                    vertex_positions=transformed_verts,
                    triangle_indices=BASE_FACE_TRIANGLES,
                    vertex_colors=np.array(
                        [face_data["color"]] * BASE_FACE_VERTICES.shape[0],
                        dtype=np.uint8,
                    ),
                    vertex_normals=face_vertex_normals,
                )

        return ret

    def rotate_rubiks_face(self, face_designator: str, angle_rad: float):
        """
        Rotates one of the 6 main faces of the Rubik's cube (e.g., U, D, F, B, L, R).
        This updates the orientations and positions of the 4 affected cubelets.

        Parameters:
        - face_designator (str): One of "U", "D", "F", "B", "L", "R".
                                 U: Up (+Y), D: Down (-Y),
                                 F: Front (+Z), B: Back (-Z),
                                 L: Left (-X), R: Right (+X).
        - angle_rad (float): Angle of rotation in radians (e.g., np.pi/2 for 90 degrees).
                             Positive angle usually means clockwise when looking at the face from outside.
        """
        # Axis of rotation in Rubik's cube's frame
        # Layer selection: (axis_idx, id_value_for_layer)
        #   axis_idx: 0 for X, 1 for Y, 2 for Z
        #   id_value_for_layer: -1 or 1, indicating which layer of cubelets
        # Rotation axis vector in Rubik's cube frame
        # Convention: For U (+Y face), rotate around +Y axis. For D (-Y face), also rotate around +Y.
        # Adjust sign of angle for "negative" faces if needed, or ensure axis points correctly.
        # Let's define rotation axis and which cubelets are selected:

        # (axis_vector_for_rotation, selection_axis_idx, selection_layer_value)
        rotation_params = {
            "U": ([0, 1, 0], 1, 1),  # Rotate around +Y, select cubelets with iy_id = 1
            "D": (
                [0, 1, 0],
                1,
                -1,
            ),  # Rotate around +Y, select cubelets with iy_id = -1
            "F": ([0, 0, 1], 2, 1),  # Rotate around +Z, select cubelets with iz_id = 1
            "B": (
                [0, 0, 1],
                2,
                -1,
            ),  # Rotate around +Z, select cubelets with iz_id = -1
            "L": (
                [1, 0, 0],
                0,
                -1,
            ),  # Rotate around +X, select cubelets with ix_id = -1
            "R": ([1, 0, 0], 0, 1),  # Rotate around +X, select cubelets with ix_id = 1
        }
        if face_designator not in rotation_params:
            print(f"Warning: Unknown face designator '{face_designator}'")
            return

        axis_vec, sel_axis_idx, sel_layer_val = rotation_params[face_designator]

        if face_designator == "U" or face_designator == "R" or face_designator == "F":
            angle_rad = -angle_rad

        orientation_matrix_for_cubelets = t3d.axangles.axangle2mat(
            np.array(axis_vec, dtype=float), angle_rad
        )

        for cubelet in self.cubelets:
            current_coord_of_interest = cubelet["current_position_in_rubiks"][
                sel_axis_idx
            ]
            # Check if the cubelet's current position along the selection axis matches the layer
            # A small tolerance might be needed for floating point comparisons if positions were not exact multiples
            # but since they are set up as +/- spacing_center, np.sign should be robust.
            if np.sign(current_coord_of_interest) == sel_layer_val:
                # This cubelet is part of the face being rotated.

                # Rotate cubelet's orientation
                R_old = cubelet["current_orientation_in_rubiks"]
                R_new = orientation_matrix_for_cubelets @ R_old
                cubelet["current_orientation_in_rubiks"] = R_new

                # Rotate cubelet's position (around Rubik's cube origin)
                p_old = cubelet["current_position_in_rubiks"]
                p_new = orientation_matrix_for_cubelets @ p_old
                cubelet["current_position_in_rubiks"] = p_new


# --- Animation Helper ---


def parse_move_string(move_str: str) -> tuple[str | None, float, int]:
    """
    Parses a Rubik's Cube move string (e.g., "U", "R'", "F2").

    Parameters:
    - move_str (str): The move string.
        - A single letter (U, D, F, B, L, R) means a 90-degree clockwise turn of that face.
        - Letter + ' (apostrophe) means a 90-degree counter-clockwise turn.
        - Letter + 2 means a 180-degree turn (conventionally clockwise).
        - Letter + 2' means a 180-degree turn (conventionally counter-clockwise).

    Returns:
    - tuple[str | None, float, int]:
        - face_designator (str | None): The face ("U", "D", "F", "B", "L", "R"). None if invalid.
        - total_angle_rad (float): Total rotation angle in radians.
                                   The angle is determined such that it produces the standard
                                   Rubik's cube notation result (CW, CCW) when applied to
                                   the face's rotation axis defined in `rotate_rubiks_face`.
        - num_90_deg_turns (int): Number of 90-degree segments (1 for 90 deg, 2 for 180 deg).
    """
    if not move_str:
        return None, 0.0, 0

    face_designator = move_str[0].upper()
    if face_designator not in ["U", "D", "F", "B", "L", "R"]:
        print(f"Warning: Invalid face designator in move '{move_str}'")
        return None, 0.0, 0

    # Angle for a 90-degree clockwise turn of the face,
    # when that face's rotation is applied around the positive axis defined in rotation_params.
    # Standard CW definition (when looking AT the face):
    # U (+Y face, rotates around +Y axis): CW is a negative angle around +Y.
    # D (-Y face, rotates around +Y axis): CW is a positive angle around +Y.
    # F (+Z face, rotates around +Z axis): CW is a negative angle around +Z.
    # B (-Z face, rotates around +Z axis): CW is a positive angle around +Z.
    # R (+X face, rotates around +X axis): CW is a negative angle around +X.
    # L (-X face, rotates around +X axis): CW is a positive angle around +X.

    actual_angle_for_this_face_CW_turn = np.pi / 2

    if len(move_str) == 1:  # e.g., "U" -> CW turn
        return face_designator, actual_angle_for_this_face_CW_turn, 1
    elif len(move_str) == 2:
        if move_str[1] == "'":  # e.g., "U'" -> CCW turn
            return face_designator, -actual_angle_for_this_face_CW_turn, 1
        elif move_str[1] == "2":  # e.g., "U2" -> 180 deg CW turn
            return face_designator, 2 * actual_angle_for_this_face_CW_turn, 2
        else:
            print(f"Warning: Invalid modifier in move '{move_str}'")
            return None, 0.0, 0
    elif len(move_str) == 3:
        if move_str[1:3] == "2'":  # e.g., "U2'" -> 180 deg CCW turn
            return face_designator, 2 * (-actual_angle_for_this_face_CW_turn), 2
        else:
            print(f"Warning: Invalid modifier in move '{move_str}'")
            return None, 0.0, 0
    else:
        print(f"Warning: Invalid move string '{move_str}'")
        return None, 0.0, 0


def visualize_rubiks_cube_animation(
    animation_data: dict,
    animation_fps: int = 120,
    cubelet_size: float = 1.0,
    gap: float = 0.02,
    cube_name: str = "RubiksCube222",
    time_offset_sec: float = 0.0,
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
    - time_offset_sec (float): Offset added to the Rerun timeline time (in seconds).
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

    # --- Log the initial rest pose at the starting timeline time ---
    rr.set_time("stable_time", duration=float(time_offset_sec))
    rubiks_cube.log_to_rerun(
        overall_translation=all_global_translations[0],
        overall_orientation_mat=all_global_orientations_mat[0],
    )
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

    print(
        f"Starting Rubik's Cube animation with {total_animation_frames} frames at {animation_fps} FPS."
    )

    # Main animation loop
    for i_frame in range(total_animation_frames):
        current_time_sec = (i_frame) / float(animation_fps) + float(time_offset_sec)
        # rr.set_time_seconds("timestamp", current_time_sec) # Changed back
        rr.set_time("stable_time", duration=current_time_sec)

        # Apply incremental rotation for the current frame if specified
        face_designator = face_designators_per_frame[i_frame]
        incremental_angle_rad = angles_per_frame[i_frame]

        if face_designator and incremental_angle_rad != 0.0:
            # print(f"Frame {i_frame} (t={current_time_sec:.3f}s): Rotating {face_designator} by {incremental_angle_rad:.4f} rad")
            rubiks_cube.rotate_rubiks_face(face_designator, incremental_angle_rad)

        current_translation = all_global_translations[i_frame]
        current_orientation = all_global_orientations_mat[i_frame]

        rubiks_cube.log_to_rerun(
            overall_translation=current_translation,
            overall_orientation_mat=current_orientation,
        )

    print(
        f"Animation processing complete. Total duration: {total_animation_frames / float(animation_fps):.2f}s. Check Rerun."
    )


def main():
    """
    Main function to demonstrate the Rubik's Cube visualization
    by providing example move sequences and global transformations,
    now using per-frame rotation specifications packaged into a dictionary.
    """
    ANIMATION_FPS = 120

    conceptual_moves = [
        {"move_str": "U", "duration_sec": 0.8},
        {"move_str": "R", "duration_sec": 0.5},
        {"move_str": "U'", "duration_sec": 0.8},
        {"move_str": "L'", "duration_sec": 1.0},
        {"move_str": "F2", "duration_sec": 1.2},
        {"move_str": "B", "duration_sec": 1.0},
        {"move_str": "D'", "duration_sec": 0.7},
        {
            "move_str": "R2",
            "duration_sec": 1.0,
        },
        {"move_str": "U2'", "duration_sec": 1.0},
    ]

    # --- Validate conceptual_moves and calculate total duration ---
    if not conceptual_moves:
        raise ValueError("Conceptual moves list is empty. Cannot generate animation.")

    for move_def in conceptual_moves:
        duration_sec = move_def.get("duration_sec", 0)
        if duration_sec <= 0:
            raise ValueError(
                f"Move '{move_def.get('move_str', 'Unknown')}' has non-positive duration_sec = {duration_sec}. "
                "All move durations must be positive."
            )

    total_animation_duration_sec = sum(
        move["duration_sec"] for move in conceptual_moves
    )

    total_frames = int(round(total_animation_duration_sec * ANIMATION_FPS))

    if total_frames == 0:
        # This implies that sum of (duration_sec * ANIMATION_FPS) rounds to 0.
        # Since durations are positive, this means they are too small for the given FPS.
        raise ValueError(
            f"Total animation duration ({total_animation_duration_sec:.3f}s) is too short "
            f"for the given FPS ({ANIMATION_FPS}), resulting in 0 total frames. "
            "Increase move durations or decrease FPS."
        )

    # --- Generate per-frame incremental rotations as NumPy arrays ---
    example_face_designators = np.full(
        total_frames, None, dtype=object
    )  # Stores string or None
    example_rotation_angles = np.zeros(total_frames, dtype=float)

    current_frame_index = 0
    for move_def in conceptual_moves:
        move_str = move_def["move_str"]
        duration_sec = move_def["duration_sec"]  # Known to be > 0

        face_designator, total_angle_rad, _ = parse_move_string(move_str)

        if face_designator is None or total_angle_rad == 0:
            # This is a pause (e.g. invalid move string, or a move that results in net zero rotation like F F')
            # or a move explicitly defined with zero rotation.
            # Its duration still contributes to the timeline.
            num_frames_for_this_pause = int(round(duration_sec * ANIMATION_FPS))
            # These frames are intentionally left with None designator and 0.0 angle by default.
            current_frame_index += num_frames_for_this_pause
        else:
            # This is an actual rotational move.
            num_frames_for_this_move = int(round(duration_sec * ANIMATION_FPS))

            if num_frames_for_this_move == 0:
                # duration_sec > 0, total_angle_rad != 0, but num_frames_for_this_move is 0.
                # This means the duration is too short for the FPS to represent this rotation.
                raise ValueError(
                    f"Move '{move_str}' (angle: {total_angle_rad:.3f} rad) has duration {duration_sec}s, "
                    f"which results in 0 frames at {ANIMATION_FPS} FPS. "
                    "Increase duration for this move, decrease FPS, or make it a zero-angle pause."
                )

            angle_per_frame = total_angle_rad / num_frames_for_this_move
            for i in range(num_frames_for_this_move):
                frame_to_apply = current_frame_index + i
                if frame_to_apply < total_frames:
                    example_face_designators[frame_to_apply] = face_designator
                    example_rotation_angles[frame_to_apply] = angle_per_frame
                # else: This part of the move extends beyond total_frames due to rounding.
                # Silently truncate, as arrays are sized by total_frames.
            current_frame_index += num_frames_for_this_move

        # Clamp current_frame_index to ensure it does not exceed total_frames,
        # which could happen due to accumulation of rounding differences between
        # sum of individual move frames and the globally calculated total_frames.
        current_frame_index = min(current_frame_index, total_frames)

    # --- Define global translation and orientation for each frame ---
    # total_frames is guaranteed to be > 0 at this point.
    example_translations = np.zeros((total_frames, 3))
    example_orientations = np.eye(3) * np.ones(
        (total_frames, 3, 3)
    )  # Initialize correctly

    for i in range(total_frames):
        # example_orientations[i] = np.eye(3) # Will be set by Rz @ Rx
        t_norm = i / (total_frames - 1) if total_frames > 1 else 0.0
        example_translations[i, 0] = 0.2 * np.sin(2 * np.pi * t_norm)
        example_translations[i, 1] = 0.1 * np.cos(4 * np.pi * t_norm)
        example_translations[i, 2] = 0.05 * np.sin(6 * np.pi * t_norm)

        angle_z = 0.5 * np.pi * t_norm
        angle_x = 0.2 * np.pi * np.sin(2 * np.pi * t_norm)
        Rz = t3d.axangles.axangle2mat([0, 0, 1], angle_z)
        Rx = t3d.axangles.axangle2mat([1, 0, 0], angle_x)
        example_orientations[i] = Rz @ Rx

    # --- Pack everything into the animation_config dictionary ---
    # example_face_designators = np.full((total_frames,), None, dtype=object)
    animation_data_dict = {
        "translations": example_translations,
        "orientations": example_orientations,
        "face_designators": example_face_designators,
        "rotation_angles": example_rotation_angles,
    }
    for i in range(total_frames):
        print(
            f"Frame {i}: {example_face_designators[i]} {example_rotation_angles[i] * 180 / np.pi}"
        )

    # Call the visualization function
    init_rerun("RubiksCube222_Demo", save=False)
    visualize_rubiks_cube_animation(
        animation_data=animation_data_dict,
        animation_fps=ANIMATION_FPS,
        cubelet_size=0.1,  # Specific value from original example
        gap=0.0005,  # Specific value from original example
    )

    print(
        "Demonstration complete using dictionary config and single face rotation per frame. "
        "Check Rerun."
    )


if __name__ == "__main__":
    main()
