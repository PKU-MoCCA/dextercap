import math

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def filter_valid_points(pts, obj_positions, invalid_val=-1000):
    """
    Filter out invalid points from the input point cloud and corresponding object positions.

    Parameters:
        pts (np.ndarray, shape=(N, D)): Input point cloud, where N is the number of points and D is the dimension (typically 3).
        obj_positions (np.ndarray, shape=(N, D)): Theoretical object positions corresponding to each point.
        invalid_val (float, optional): Value indicating invalid points. Default is -1000.

    Returns:
        valid_pts (np.ndarray, shape=(M, D)): Valid points after filtering, where M <= N.
        valid_theory (np.ndarray, shape=(M, D)): Corresponding valid theoretical positions.
    """
    # Create a mask for valid points (not equal to invalid_val in any dimension)
    valid_mask = np.all(pts != invalid_val, axis=-1)
    valid_pts = pts[valid_mask]
    valid_theory = obj_positions[valid_mask]
    return valid_pts, valid_theory


def compute_initial_pose(observed_pts, theory_pts):
    """
    Compute the initial pose (translation and rotation) using the Kabsch algorithm.

    Parameters:
        observed_pts (np.ndarray, shape=(N, 3)): Observed points.
        theory_pts (np.ndarray, shape=(N, 3)): Theoretical points.

    Returns:
        t (np.ndarray, shape=(3,)): Translation vector.
        quat (np.ndarray, shape=(4,)): Quaternion representing rotation (x, y, z, w).
    """
    # Compute centroids
    obs_centroid = observed_pts.mean(axis=0)
    theory_centroid = theory_pts.mean(axis=0)
    # Center the points
    obs_centered = observed_pts - obs_centroid
    theory_centered = theory_pts - theory_centroid

    # SVD for optimal rotation
    H = theory_centered.T @ obs_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = obs_centroid - R @ theory_centroid
    quat = Rotation.from_matrix(R).as_quat()
    return t, quat


def pose_optimization(
    params, observed_pts, theory_pts, lambda_t=0.1, lambda_r=0.1, prev_pose=None
):
    """
    Objective function for pose optimization.

    Parameters:
        params (np.ndarray, shape=(7,)): Concatenated translation (3,) and quaternion (4,).
        observed_pts (np.ndarray, shape=(N, 3)): Observed points.
        theory_pts (np.ndarray, shape=(N, 3)): Theoretical points.
        lambda_t (float): Weight for translation regularization.
        lambda_r (float): Weight for rotation regularization.
        prev_pose (tuple or None): Previous pose (translation, quaternion) for temporal regularization.

    Returns:
        error (float): Optimization loss value.
    """
    center = params[:3]
    quat = params[3:7]
    quat /= np.linalg.norm(quat)  # normalize quaternion
    R = Rotation.from_quat(quat).as_matrix()

    # Transform theory points to observed space
    transformed_theory = (R @ theory_pts.T).T + center

    # Compute mean squared error
    error = np.sum((observed_pts - transformed_theory) ** 2)
    point_num = observed_pts.shape[0]
    if point_num != 0:
        error /= point_num

    # Temporal regularization (commented out)
    # if prev_pose is not None:
    #     prev_center, prev_quat = prev_pose
    #     delta_center = np.sum((center - prev_center)**2)
    #     delta_rot = Rotation.from_quat(quat).inv() * Rotation.from_quat(prev_quat)
    #     angle = delta_rot.magnitude()
    #     error += lambda_t * delta_center + lambda_r * angle**2

    return error


def process_frame(frame_pts, obj_positions, prev_pose=None):
    """
    Process a single frame to estimate the object's pose.

    Parameters:
        frame_pts (np.ndarray, shape=(N, 3)): Observed points for the frame.
        obj_positions (np.ndarray, shape=(N, 3)): Theoretical object positions.
        prev_pose (tuple or None): Previous pose (translation, quaternion) for initialization.

    Returns:
        opt_center (np.ndarray, shape=(3,)): Optimized translation vector.
        opt_quat (np.ndarray, shape=(4,)): Optimized quaternion (x, y, z, w).
        nit (int): Number of iterations performed by the optimizer.
        loss (float): Final loss value.
    """
    # Filter valid points
    valid_pts, valid_theory = filter_valid_points(frame_pts, obj_positions)

    # Initialize pose
    if prev_pose is None:  # initial frame
        initial_center, initial_quat = compute_initial_pose(valid_pts, valid_theory)
    else:
        initial_center, initial_quat = prev_pose

    initial_params = np.concatenate([initial_center, initial_quat])

    # Optimize pose
    result = minimize(
        pose_optimization,
        initial_params,
        args=(valid_pts, valid_theory, 0.1, 0.001, None),
        method="L-BFGS-B",
        options={"maxiter": 100},
    )

    opt_params = result.x
    opt_center = opt_params[:3]
    opt_quat = opt_params[3:7]
    opt_quat /= np.linalg.norm(opt_quat)

    return opt_center, opt_quat, result.nit, result.fun


def main(pts, obj_positions, discontinuity=[], start=0):
    """
    Main function to process all frames and estimate poses.

    Parameters:
        pts (np.ndarray, shape=(F, N, 3)): All observed points for F frames.
        obj_positions (np.ndarray, shape=(N, 3)): Theoretical object positions.
        discontinuity (list, optional): List of frame indices where discontinuities occur.
        start (int, optional): Starting frame index.

    Returns:
        poses (np.ndarray, shape=(F, 7)): Estimated poses for all frames (translation + quaternion).
    """
    num_frames = pts.shape[0]
    poses = []
    discontinuity.insert(0, start)
    discontinuity.append(num_frames)
    nits, loss_distance, loss_t, loss_q = [], [], [], []

    for i in range(len(discontinuity) - 1):
        start = discontinuity[i]
        end = discontinuity[i + 1]
        print(f"Processing frame {start} to {end}")
        prev_pose = None

        for j in range(start, end):
            if j % 500 == 0:
                print(f"Processing frame {j}/{num_frames}")
            center, quat, nit, loss = process_frame(
                pts[j], obj_positions, prev_pose=prev_pose
            )

            nits.append(nit)
            loss_distance.append(math.sqrt(loss))

            if prev_pose is not None:
                prev_center, prev_quat = prev_pose
                delta_center = math.sqrt(np.sum((center - prev_center) ** 2))  # 1e-2
                delta_rot = Rotation.from_quat(quat).inv() * Rotation.from_quat(
                    prev_quat
                )
                angle = delta_rot.magnitude()  # rad, 1e-1
                # Take the absolute value of the angle in radians and convert to degrees
                angle = math.fabs(angle) * 180 / math.pi
                loss_t.append(delta_center)
                loss_q.append(angle)
            else:
                loss_t.append(0)
                loss_q.append(0)

            prev_pose = (center, quat)
            params = np.concatenate([center, quat])
            poses.append(params)

    nits = np.array(nits)
    loss_distance = np.array(loss_distance)
    loss_t = np.array(loss_t)
    loss_q = np.array(loss_q)

    print(f"nits: mean: {nits.mean()}, std: {nits.std()}")
    print(
        f"loss_distance: mean: {1000 * loss_distance.mean()} mm, std: {1000 * loss_distance.std()} mm"
    )
    print(
        f"translation: mean: {1000 * loss_t.mean()} mm, std: {1000 * loss_t.std()} mm"
    )
    print(f"rotation: mean: {loss_q.mean()} deg, std: {loss_q.std()} deg")

    return np.array(poses)


if __name__ == "__main__":
    pts = np.load("object/0414/pts_obj.npy")
    print(pts.shape)
    obj_positions = np.load("object/0414/cube_init.npy")
    print(obj_positions.shape)
    poses = main(pts, obj_positions, start=1334)
    print(poses.shape)
    np.save("object/0414/obj_poses.npy", poses)
