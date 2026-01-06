import os

import numpy as np
import rerun as rr
import smplx
import torch
import trimesh
from scipy.spatial.transform import Rotation as R

import config
from Utility.utils_mesh import compute_vertex_normals

bone_length_with_root = np.array(
    [
        [0, 1, 81],
        [1, 2, 30],
        [2, 3, 25],
        [0, 4, 77],
        [4, 5, 33],
        [5, 6, 30],
        [0, 10, 69],
        [10, 11, 36],
        [11, 12, 29],
        [0, 7, 68],
        [7, 8, 28],
        [8, 9, 22],
        [0, 13, 45],
        [13, 14, 34],
        [14, 15, 33],
    ]
)
bone_length_without_root = np.array(
    [
        [1, 2, 30],
        [2, 3, 25],
        [4, 5, 33],
        [5, 6, 30],
        [10, 11, 36],
        [11, 12, 29],
        [7, 8, 28],
        [8, 9, 22],
        [13, 14, 34],
        [14, 15, 33],
    ]
)
iterations = 2000

smplx_model = smplx.create(
    model_path=config.data["smpl_model_path"],
    model_type="mano",
    flat_hand_mean=True,
    is_rhand=False,
    use_pca=False,
    batch_size=1,
)


def fit_mano_shape():
    hand_shape = torch.zeros((1, 10), requires_grad=True)

    optimizer = torch.optim.Adam([hand_shape], lr=1e-3)

    for i in range(iterations):
        optimizer.zero_grad()
        hand_parms = {
            "global_orient": torch.zeros(1, 3),
            "transl": torch.zeros(1, 3),
            "hand_pose": torch.zeros(1, 45),
            "betas": hand_shape,
        }
        smplx_output = smplx_model(**hand_parms)
        joints = smplx_output.joints[0]
        bone_lengths = bone_length_without_root
        start_index = bone_lengths[:, 0]
        end_index = bone_lengths[:, 1]
        target_length = torch.tensor(bone_lengths[:, 2] * 1e-3, requires_grad=False)
        loss = torch.norm(
            torch.norm(joints[end_index] - joints[start_index], dim=1) - target_length
        )
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Iteration {i}, loss: {loss.item()}")

    return hand_shape.detach().cpu().numpy()


def main():
    hand_parms = {
        "global_orient": torch.zeros(1, 3),
        "transl": torch.zeros(1, 3),
        "hand_pose": torch.zeros(1, 45),
        "betas": torch.zeros(1, 10),
    }
    smplx_output = smplx_model(**hand_parms)

    vertices = smplx_output.vertices[0].detach().cpu().numpy()
    joints = smplx_output.joints[0].detach().cpu().numpy()

    # Record root position
    rr.init("MANO_shape_fit", spawn=True)
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # Set an up-axis = +Y
    rr.log(
        "hand/original",
        rr.Mesh3D(
            vertex_positions=vertices,
            triangle_indices=smplx_model.faces,
            vertex_normals=compute_vertex_normals(vertices, smplx_model.faces),
        ),
    )
    rr.log(
        "hand/original_joints",
        rr.Points3D(
            positions=joints, radii=0.001, colors=np.array([[255, 0, 0]] * len(joints))
        ),
    )

    # hand_shape = fit_mano_shape()
    hand_shape = np.array(
        [
            [
                0.2562402,
                -0.46635702,
                -1.6472689,
                0.1752234,
                -0.0941459,
                0.91344696,
                0.21890263,
                1.025448,
                0.03658989,
                -0.2983594,
            ]
        ],
        dtype=np.float32,
    )

    print(hand_shape)
    np.save("hand_shape.npy", hand_shape[0])

    hand_parms["transl"][0, 2] = 0.17
    hand_parms["betas"] = torch.tensor(hand_shape)
    smplx_output = smplx_model(**hand_parms)
    vertices = smplx_output.vertices[0].detach().cpu().numpy()
    joints = smplx_output.joints[0].detach().cpu().numpy()
    rr.log(
        "hand/fitted",
        rr.Mesh3D(
            vertex_positions=vertices,
            triangle_indices=smplx_model.faces,
            vertex_normals=compute_vertex_normals(vertices, smplx_model.faces),
        ),
    )
    rr.log(
        "hand/fitted_joints",
        rr.Points3D(
            positions=joints, radii=0.001, colors=np.array([[0, 255, 0]] * len(joints))
        ),
    )

    obj_path = os.path.join(os.path.dirname(__file__), "hand_model.obj")
    mesh = trimesh.load(obj_path)
    rot = R.from_rotvec(np.array([0, np.pi / 2, 0])) * R.from_rotvec(
        np.array([np.pi / 2, 0, 0])
    )
    mesh.vertices = rot.apply(mesh.vertices)
    mesh.vertices += np.array([0.25, 0, 0.6])

    rr.log(
        "hand/obj",
        rr.Mesh3D(
            vertex_positions=mesh.vertices,
            triangle_indices=mesh.faces,
            vertex_normals=compute_vertex_normals(mesh.vertices, mesh.faces),
        ),
    )


if __name__ == "__main__":
    main()
