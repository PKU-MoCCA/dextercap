# Hand Reconstruction Part

Fit 3D mocap marker points to the MANO hand model.

This module starts from **3D points** (already triangulated).

## Description

1. Given `N` hand mocap points.
   1. Mocap points must keep a consistent order across frames.
   2. `config.data["body_part_index_path"]` provides a per-point segment id (used to decide which MANO segment each marker corresponds to).
   3. Run `python HandReconstruction/Data/mano_segment.py` to visualize the segments on the MANO mesh.
2. Each frame is optimized independently using `PyTorch`.
   1. The optimization process involves adjusting the MANO parameters: translation, orientation, and pose.
   2. The optimization is initialized with default parameters.
   3. For each following frame, the optimization is initialized with the previous frame's parameters.
3. During the `PyTorch` optimization, the `MANO pose` parameters are regularized to avoid unrealistic hand poses. Please reference `HandReconstruction/config.py` to see the regularization details.
   1. If `config.hand["dof"][i]` is `1`, the `i`-th joint is optimized only with `z` axis rotation, a.k.a. bending.
   2. If `config.hand["dof"][i]` is `2`, the `i`-th joint is optimized with `y` and `z` axis rotation, a.k.a. bending and spreading.
   3. If `config.hand["dof"][i]` is `3`, the `i`-th joint is optimized with `x`, `y`, and `z` axis rotation, a.k.a. bending, spreading, and twisting.

## Convention

> [!IMPORTANT] 
> The DOF are defined in the order of index finger, middle finger, pinky finger, ring finger, thumb, not the order of thumb, index finger, middle finger, ring finger, pinky finger.
> Root joint is not included in the DOF list!

1. The `MANO` model has several degrees of freedom (DOF) for the fingers.
   1. Defined in `config.py: hand["dof"]`
   2. The DOF are biologically inspired. Original `MANO` model has coordinate system aligned with the world coordinate system, but we use the coordinate system aligned with the bone, so the DOF angles means different things.
   3. In the order of right hand:
   ```
      15 - 14 - 13 - \ Thumb
                      \
      3-- 2 -- 1 ----- 0 Index
       6 -- 5 -- 4 -- / Middle
   12 -- 11 -- 10 -- / Ring
     9 -- 8 -- 7 -- / Pinky
   ```

## Usage

1. Configure the paths and parameters in `HandReconstruction/config.py`.

   - `data["hand_mocap_point_data_path"]`: `pts_hand*.npy` with shape `[num_frames, num_points, 3]` (meters).
     Invalid points should be set to `data["invalid_point_value"]` (default: `[-1000, -1000, -1000]`).
   - `data["body_part_index_path"]`: `idx_hand*.npy` with shape `[num_points]` (integer segment id per marker).
   - Adjust optimization hyperparameters (iterations, learning rate, regularization, etc.) in `optimize`.

2. Run the optimization, generate the loss curve, (or visualize the results instantly if you set `optimize["test_mode"]` to `True`)

   ```shell
   python -m HandReconstruction.main
   ```

3. Outputs are written to `HandReconstruction/Result/<mocap_session_name>/<timestamp>/frame_*.npz` (chunked).
