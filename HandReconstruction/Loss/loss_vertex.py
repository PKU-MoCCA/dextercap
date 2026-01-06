import numpy as np
import torch

from HandReconstruction.Utility.utils_mesh import calculate_nearest_point


def loss_vertex_vanilla_calibration(
    vertices, vertice_of_joint, target_points, mocap_point_index
):
    """
    Calculate vertex loss during calibration phase using part-to-part correspondence.

    Parameters:
    - vertices (torch.Tensor, shape=(num_vertices, 3)): MANO mesh vertices in world coordinate.
    - vertice_of_joint (list, shape=(num_of_this_joint_vertices,)): List of vertex indices corresponding to a specific joint.
    - target_points (torch.Tensor, shape=(num_points, 3)): Target mocap points in world coordinate.
    - mocap_point_index (torch.Tensor, shape=(num_points,)): Indices of mocap points for the current joint.

    Returns:
    - loss (torch.Tensor, shape=(1,)): Sum of minimum distances between joint vertices and target points.
    """
    dist = torch.cdist(vertices[vertice_of_joint], target_points[mocap_point_index], 2)
    return torch.min(dist, dim=0)[0].sum()


def loss_vertex_meshdis_calibration(
    vertices,
    vertice_of_joint,
    mesh_faces,
    target_points,
    mocap_point_index,
    calibration_result,
):
    """
    Calculate vertex loss during calibration phase using meshdis calibration.

    Args:
        - vertices (torch.Tensor, shape=(num_vertices, 3)): MANO mesh vertices in world coordinate.
        - vertice_of_joint (torch.Tensor, shape=(num_vertices_in_this_body,)): Vertex indices corresponding to a specific joint.
        - mesh_faces (torch.Tensor, shape=(num_faces, 3)): Faces of the mesh.
        - target_points (torch.Tensor, shape=(num_points, 3)): Target mocap points in world coordinate.
        - mocap_point_index (torch.Tensor, shape=(num_points,)): Indices of mocap points for the current joint.
        - calibration_result ([torch.Tensor, torch.Tensor], shape=[(num_points, 3), (num_points, 3)]): Calibration barycentric indices and barycentric coordinates.

    Returns:
        - loss (torch.Tensor): Sum of minimum distances between nearest points on the mesh and target points.
        - updated_calibration_result ([np.ndarray, np.ndarray], shape=[(num_points, 3), (num_points, 3)]): Updated calibration barycentric indices and barycentric coordinates.
    """
    in_this_submesh_mask = torch.isin(mesh_faces, vertice_of_joint)
    submesh_face_mask = torch.all(in_this_submesh_mask, dim=1)
    submesh_faces = mesh_faces[submesh_face_mask]
    nearest_distance, nearest_point_barycentric_index, nearest_point_barycentric = (
        calculate_nearest_point(
            target_points[mocap_point_index], vertices, submesh_faces
        )
    )

    # Create copy of calibration results
    updated_calibration_result = [
        np.copy(calibration_result[0]),
        np.copy(calibration_result[1]),
    ]

    ##########################################################
    # Postpositional calibrations take precedence
    ##########################################################
    updated_calibration_result[0][mocap_point_index.cpu().clone().detach().numpy()] = (
        nearest_point_barycentric_index
    )
    updated_calibration_result[1][mocap_point_index.cpu().clone().detach().numpy()] = (
        nearest_point_barycentric.cpu().clone().detach().numpy()
    )
    ##########################################################
    # Postpositional calibrations take precedence
    ##########################################################

    ##########################################################
    # Earlier calibrations take precedence
    ##########################################################
    # # Only update points that haven't been calibrated yet (all coordinates are 0)
    # current_coords = calibration_result[1][mocap_point_index.cpu().clone().detach().numpy()]
    # uncalibrated_mask = np.all(current_coords == 0, axis=1) # (mocap_point_num_in_this_body, )
    # uncalibrated_point_indices = mocap_point_index.cpu().clone().detach().numpy()[uncalibrated_mask] # (uncalibrated_mocap_point_num_in_this_body, ), this index is of the total 600 mocap points

    # # Update only uncalibrated points
    # if len(uncalibrated_point_indices) > 0:
    #     updated_calibration_result[0][uncalibrated_point_indices] = nearest_point_barycentric_index[uncalibrated_mask]
    #     updated_calibration_result[1][uncalibrated_point_indices] = nearest_point_barycentric[uncalibrated_mask]
    ##########################################################
    # Earlier calibrations take precedence
    ##########################################################

    return nearest_distance.sum(), updated_calibration_result


def loss_vertex_vanilla_inference(
    vertices, vertice_of_joint, target_points, mocap_point_mask, calibration_result
):
    """
    Calculate vertex loss during inference phase using point-to-point correspondence.

    Parameters:
    - vertices (torch.Tensor, shape=(num_vertices, 3)): MANO mesh vertices in world coordinate.
    - vertice_of_joint (list): List of vertex indices corresponding to a specific joint.
    - target_points (torch.Tensor, shape=(num_points, 3)): Target mocap points in world coordinate.
    - mocap_point_mask (torch.Tensor, shape=(num_points,)): Boolean mask indicating valid mocap points for current joint.
    - calibration_result (torch.Tensor, shape=(num_points, num_vertices)): Calibration weights matrix.

    Returns:
    - loss (torch.Tensor): Combined loss from part-to-part and point-to-point correspondences.
    """
    calibration_value = torch.max(calibration_result, dim=1)[0]
    calibrated_mask = ~torch.isclose(calibration_value, torch.tensor(0.0))
    this_body_calibrated_mask = mocap_point_mask & calibrated_mask
    this_body_not_calibrated_mask = mocap_point_mask & ~calibrated_mask
    this_body_nearest_mano_vertex_mask = torch.argmax(
        calibration_result[this_body_calibrated_mask], dim=1
    )

    dist_part_to_part = torch.cdist(
        vertices[vertice_of_joint], target_points[this_body_not_calibrated_mask], 2
    )
    dist_point_to_point = torch.norm(
        vertices[this_body_nearest_mano_vertex_mask]
        - target_points[this_body_calibrated_mask],
        dim=1,
    )
    inference_loss = (
        torch.min(dist_part_to_part, dim=0)[0].sum() + dist_point_to_point.sum()
    )
    return inference_loss


def loss_vertex_meshdis_inference(
    vertices, vertice_of_joint, target_points, mocap_point_mask, calibration_result
):
    """
    Calculate vertex loss during inference phase using point-to-point correspondence.

    Parameters:
    - vertices (torch.Tensor, shape=(num_vertices, 3)): MANO mesh vertices in world coordinate.
    - vertice_of_joint (torch.Tensor, shape=(num_vertices_in_this_body,)): Vertex indices corresponding to a specific joint.
    - target_points (torch.Tensor, shape=(num_points, 3)): Target mocap points in world coordinate.
    - mocap_point_mask (torch.Tensor): Boolean mask indicating valid mocap points for current joint.
    - calibration_result ([torch.Tensor, torch.Tensor]): Tuple of barycentric indices and coordinates.

    Returns:
    - loss (torch.Tensor): Combined loss from part-to-part and point-to-point correspondences.
    """
    device = vertices.device

    # Unpack the calibration result
    barycentric_indices, barycentric_coords = calibration_result
    barycentric_indices = torch.tensor(
        barycentric_indices, dtype=torch.int32, device=device
    )
    barycentric_coords = torch.tensor(
        barycentric_coords, dtype=torch.float32, device=device
    )

    calibrated_mask = ~torch.isclose(
        barycentric_coords, torch.tensor(0.0, dtype=torch.float32, device=device)
    ).all(dim=1)
    this_body_calibrated_mask = mocap_point_mask & calibrated_mask
    this_body_not_calibrated_mask = mocap_point_mask & ~calibrated_mask

    dist_part_to_part = torch.cdist(
        vertices[vertice_of_joint], target_points[this_body_not_calibrated_mask], 2
    )

    if torch.any(this_body_calibrated_mask):
        triangle_vertices = vertices[barycentric_indices[this_body_calibrated_mask]]
        interpolated_points = (
            triangle_vertices
            * barycentric_coords[this_body_calibrated_mask].unsqueeze(-1)
        ).sum(dim=1)
        dist_point_to_point = torch.norm(
            interpolated_points - target_points[this_body_calibrated_mask], dim=1
        )
    else:
        dist_point_to_point = torch.tensor(0.0)

    inference_loss = (
        torch.min(dist_part_to_part, dim=0)[0].sum() + dist_point_to_point.sum()
    )
    return inference_loss
