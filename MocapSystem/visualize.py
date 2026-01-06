import argparse
import os
from typing import List

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import utils


def init_rerun(name: str, camera_ids: List[int]):
    """
    Initialize the rerun visualization.

    Parameters:
    - name (str): The name of the rerun session.
    """
    # rr.ViewCoordinates.
    rr.init(name, spawn=True)
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)  # Set an up-axis = +Y
    rr.set_time("stable_time", duration=0)

    # # visualize axes
    # rr.log(
    #     "arrows",
    #     rr.Arrows3D(
    #         vectors=[[1,0,0], [0,1,0], [0,0,1]],
    #         radii=0.003,
    #         labels=['x', 'y', 'z']
    #     )
    # )

    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(name="3D"),
        rrb.Vertical(
            rrb.Tabs(
                # Note that we re-project the annotations into the 2D views:
                # For this to work, the origin of the 2D views has to be a pinhole camera,
                # this way the viewer knows how to project the 3D annotations into the 2D views.
                *(
                    rrb.Spatial2DView(
                        name=f"cam_{cam_id}",
                        origin=f"camera/camera_{cam_id}",
                        contents=[
                            f"$origin/**",
                            # f"marker/**",
                        ],
                    )
                    for cam_id in camera_ids
                ),
                name="2D",
            ),
            # rrb.TextDocumentView(name="Readme"),
            # row_shares=[2, 1],
        ),
    )

    rr.send_blueprint(blueprint)


def visualize_cameras(
    camera_ids: List[int], camera_param_path: str, camera_param_fmt: str
):
    for cam_id in camera_ids:
        cam_param = np.load(
            os.path.join(camera_param_path, camera_param_fmt.format(cam_id))
        )
        instrinsic = cam_param["intrinsics"]
        tvec, rmat = cam_param["tvecs"], cam_param["rmats"]

        # rr.log(f"camera/label_{cam_id}",
        #     rr.Points3D(
        #         positions=[0,0,0],
        #         radii=0.01,
        #         labels=cam_id
        #     )
        # )
        # rr.log(
        #     f"camera/label_{cam_id}",
        #     rr.Transform3D(mat3x3=rmat, translation=tvec, from_parent=True),
        # )

        rr.log(
            f"camera/camera_{cam_id}",
            rr.Pinhole(
                image_from_camera=instrinsic,
                width=2448,
                height=2048,
                # camera_xyz=rr.ViewCoordinates.RDF
            ),
        )

        rr.log(
            f"camera/camera_{cam_id}",
            rr.Transform3D(mat3x3=rmat, translation=tvec, from_parent=True),
        )


def visualize_2d_points(
    camera_ids: List[int],
    points_2d: np.ndarray,
    edges: List[List[int]],
    color: np.ndarray | List = [60, 200, 205],
    edge_color: np.ndarray | List = [128, 128, 128],
):
    edge_indices = np.asarray(edges).reshape(-1, 2)

    num_cam, num_points = points_2d.shape[:2]
    for cam_idx, cam_id in enumerate(camera_ids):
        rr.log(
            f"camera/camera_{cam_id}/image_point_{cam_id}",
            rr.Points2D(
                positions=points_2d[cam_idx],
                radii=5,
                colors=color,
                # labels=cam_id
            ),
        )

        edge_points = points_2d[cam_idx, edge_indices].reshape(-1, 2, 2)
        edge_points[np.any(edge_points <= 0, axis=(1, 2))] = 0

        rr.log(
            f"camera/camera_{cam_id}/image_edges_{cam_id}",
            rr.LineStrips2D(
                edge_points,
                radii=1,
                colors=edge_color,
                # labels=cam_id
            ),
        )


def visualize_points(points: np.ndarray, color: np.ndarray | List = [255, 255, 60]):
    npts, ndim = points.shape
    for j in range(npts):
        rr.log(
            f"marker/marker_{j}",
            rr.Points3D(
                positions=points[j],
                radii=0.001,
                colors=color,
                # labels=body_part_indices[j]
            ),
        )


def visualize_edges(
    points: np.ndarray,
    edges: List[List[int]],
    color: np.ndarray | List = [128, 128, 128],
):
    edge_points = points[np.asarray(edges).reshape(-1, 2)].reshape(-1, 2, 3)
    edge_points[np.any(edge_points == 0, axis=(1, 2))] = 0

    rr.log("edges", rr.LineStrips3D(edge_points, radii=0.0002, colors=color))

def visualize_blocks(points, blocks):
    for label in blocks:
        marker_indices = blocks[label]["markers"]
        exist = True
        for idx in marker_indices:
            if np.any(points[idx] == 0):
                exist = False
                break

        if exist:
            center = np.mean(points[marker_indices, :], axis=0)
        else:
            center = np.zeros(3)

        rr.log(
            f"block/block_{label}",
            rr.Points3D(positions=center, radii=0.00001, labels=label),
        )


def visualize():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=str, required=True)
    parser.add_argument(
        "--camera-ids",
        default="2,3,5,6,7,8,9,11,13,14",
        type=str,
        help="id of cameras, in the fmt of 2,3,4...",
    )
    parser.add_argument(
        "--label-patches", type=str, required=True, help="label_patches.json"
    )

    parser.add_argument("--cam-param-path", type=str, required=True)
    parser.add_argument("--cam-param-fmt", type=str, default="{0}.npz")

    parser.add_argument("--fps", type=int, default=20)

    args = parser.parse_args()

    camera_ids = np.array(list(map(int, args.camera_ids.strip("()\"',").split(","))))
    print(f"camera_ids: {camera_ids}")
    fps = args.fps

    marker_defs, block_defs, patches = utils.load_label_patches(args.label_patches)
    edges = utils.get_edges_from_block_def(block_defs=block_defs)

    with np.load(args.input_file) as data:
        points_3d = data["markers_3d_positions"]  # [nframe, npts, 3]
        points_2d = data["markers_undistorted"]  # [nframe, ncam, npts, 3]

    points_3d[np.all(points_3d == [-1e3, -1e3, -1e3], axis=-1)] = 0

    invalid_points = points_2d[..., 0] < 0
    points_2d[invalid_points] = -1

    num_frames, num_cameras, num_points = points_2d.shape[:3]
    assert num_cameras == len(camera_ids)

    init_rerun("hand mocap", camera_ids=camera_ids)
    visualize_cameras(
        camera_ids=camera_ids,
        camera_param_path=args.cam_param_path,
        camera_param_fmt=args.cam_param_fmt,
    )

    for frame in range(num_frames):
        rr.set_time("stable_time", duration=frame / fps)

        visualize_points(points_3d[frame])
        visualize_2d_points(camera_ids, points_2d[frame], edges)
        visualize_edges(points_3d[frame], edges)
        visualize_blocks(points_3d[frame], block_defs)


if __name__ == "__main__":
    visualize()
