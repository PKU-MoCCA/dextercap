# Object Reconstruction

Estimate object pose (and, for Rubik's Cube, twist state) from **3D object mocap markers**.

This module starts from **3D points** (already triangulated).

## Folder layout

- `MocapData/`: example 3D mocap inputs (currently `RubiksCube_00`)
- `Result/<mocap_session_name>/config.py`: per-session metadata (used by `Dataset.generate_dataset`)
- `Result/<mocap_session_name>/<run_id>/`: reconstructed pose outputs

## Rigid objects (Cuboid / Cylinder / Ring / Plate / Prism)

For rigid objects we estimate a 6D pose per frame by aligning the observed marker cloud to a canonical marker layout
in the object coordinate system.

**Output format**
- `*.npy`: shape `[num_frames, 7]`, each row is `[x, y, z, qx, qy, qz, qw]` (meters + quaternion in xyzw).

**Reference implementation**
- `cube_reconstruction.py` shows a minimal per-frame solver:
  - filters invalid points (`[-1000, -1000, -1000]`)
  - initializes pose with the Kabsch algorithm
  - refines translation + quaternion with L-BFGS

## 2x2x2 Rubik's Cube (RubiksCube_00)

### Visualization

To try out the 2x2x2 Rubik's Cube rotation animation visualization, run:

```bash
python -m ObjectReconstruction.rubikscube
```

### Reconstruction

To try out the 2x2x2 Rubik's Cube reconstruction, run:

```bash
python -m ObjectReconstruction.rubikscube_reconstruction
```

**Inputs (real data mode)**
Configured at the top of `rubikscube_reconstruction.py`:
- `pts_obj*.npy`: shape `[num_frames, 384, 3]` (24 facelets × 16 markers), invalid marker = `[-1000, -1000, -1000]`
- `obj_face_idx*.npy`: shape `[384]`, integer in `[0, 23]` (canonical facelet id for each marker)
- `obj_point_idx*.npy`: shape `[384]`, integer in `[0, 15]` (marker id within each facelet)

> [!NOTE]
> The Kabsch algorithm requires exact point-to-point correspondence.
> The reconstruction assumes markers are ordered by **facelet id** (0→23), and within each facelet by **marker id** (0→15).
> When `use_real_data=True`, `rubikscube_reconstruction.py` uses `obj_face_idx` / `obj_point_idx` to reorder the raw
> point cloud into this canonical ordering before running alignment and twist estimation.

**Outputs**
Saved as a `.npz` containing:
- `translations`: shape `[num_frames, 3]`
- `orientations`: shape `[num_frames, 3, 3]` (rotation matrices)
- `face_designators`, `rotation_angles`: shape `[num_frames]` (discrete twist state)
