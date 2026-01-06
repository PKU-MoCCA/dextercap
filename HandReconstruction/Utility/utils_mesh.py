import numpy as np
import rerun as rr
import torch
import trimesh


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


def farthest_point_sampling(points, num_samples):
    """
    Select points from a point cloud using farthest point sampling.

    Parameters:
    - points (np.ndarray, shape=(B, N, 3) or (N, 3)): Input point cloud.
    - num_samples (int): Number of points to sample.

    Returns:
    - np.ndarray, shape=(B, num_samples, 3) or (num_samples, 3): Sampled point cloud.
    """
    if len(points.shape) == 2:
        points = np.expand_dims(points, axis=0)
    B, N, D = points.shape

    # Initialize the indices of the sampled points
    centroids = np.zeros((B, num_samples), dtype=np.int32)

    # Initialize the maximum distance array
    distances = np.ones((B, N)) * 1e10

    # Randomly select the initial point
    farthest = np.random.randint(0, N, B)
    batch_indices = np.arange(B)

    for i in range(num_samples):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest].reshape(B, 1, D)
        dist = np.sum((points - centroid) ** 2, -1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = np.argmax(distances, axis=1)

    sampled_points = points[batch_indices[:, None], centroids]
    if B == 1:
        sampled_points = sampled_points[0]
    return sampled_points


def sample_point_cloud_from_mesh(
    mesh: trimesh.Trimesh, num_points: int, oversample_factor: int = 2
):
    """
    Sample point cloud from 3D mesh by oversampling and farthest point sampling.

    Parameters:
    - mesh (trimesh.Trimesh): Input 3D mesh object.
    - num_points (int): Number of points to sample.
    - oversample_factor (int): Oversampling factor.

    Returns:
    - np.ndarray, shape=(num_points, 3): Sampled point cloud.
    """

    # Oversample the number of points
    oversample_points = num_points * oversample_factor

    # Sample points from the mesh
    points, _ = trimesh.sample.sample_surface_even(mesh, oversample_points)

    # Downsample using farthest point sampling
    selected_points = farthest_point_sampling(points, num_points)

    return selected_points


@torch.jit.script
def calculate_nearest_point(
    target_points: torch.Tensor, vertices: torch.Tensor, submesh_faces: torch.Tensor
):
    """
    Calculate nearest point on mesh for each target point.

    Parameters:
    - target_points (torch.Tensor, shape=(num_points, 3)): Target points to find nearest points for.
    - vertices (torch.Tensor, shape=(num_vertices, 3)): Mesh vertices.
    - submesh_faces (np.ndarray, shape=(num_faces, 3)): Face indices of the mesh.

    Returns:
    - torch.Tensor, shape=(num_points,): Minimum distances.
    - torch.Tensor, shape=(num_points,): Indices of nearest face vertices.
    - torch.Tensor, shape=(num_points, 3): Barycentric coordinates.
    """
    device = vertices.device
    num_points = target_points.shape[0]

    # Reshape the points into a broadcastable form
    points = target_points.unsqueeze(1)  # (num_points, 1, 3)

    # Get triangle vertices
    triangle_vertices = vertices[submesh_faces]  # (num_faces, 3, 3)

    # Calculate edges
    edge1 = triangle_vertices[:, 1, :] - triangle_vertices[:, 0, :]  # (num_faces, 3)
    edge2 = triangle_vertices[:, 2, :] - triangle_vertices[:, 0, :]  # (num_faces, 3)

    # Calculate the vector from the triangle origin to the point
    v = points - triangle_vertices[:, 0, :].unsqueeze(0)  # (num_points, num_faces, 3)

    # Calculate the dot products of the barycentric coordinates
    d00 = (edge1 * edge1).sum(dim=1).unsqueeze(0)  # (1, num_faces)
    d01 = (edge1 * edge2).sum(dim=1).unsqueeze(0)  # (1, num_faces)
    d11 = (edge2 * edge2).sum(dim=1).unsqueeze(0)  # (1, num_faces)
    d20 = (v * edge1.unsqueeze(0)).sum(dim=2)  # (num_points, num_faces)
    d21 = (v * edge2.unsqueeze(0)).sum(dim=2)  # (num_points, num_faces)

    denom = d00 * d11 - d01 * d01  # (1, num_faces)
    denom = denom + (denom == 0).float() * 1e-10  # Avoid division by zero

    v_coord = (d11 * d20 - d01 * d21) / denom  # (num_points, num_faces)
    w_coord = (d00 * d21 - d01 * d20) / denom  # (num_points, num_faces)
    u_coord = 1.0 - v_coord - w_coord  # (num_points, num_faces)

    # Determine if projections are inside the triangle
    inside_mask = (
        (u_coord >= 0) & (v_coord >= 0) & (w_coord >= 0)
    )  # (num_points, num_faces)

    # Compute nearest points for inside projections
    nearest_inside = (
        u_coord.unsqueeze(-1) * triangle_vertices[:, 0, :].unsqueeze(0)
        + v_coord.unsqueeze(-1) * triangle_vertices[:, 1, :].unsqueeze(0)
        + w_coord.unsqueeze(-1) * triangle_vertices[:, 2, :].unsqueeze(0)
    )  # (num_points, num_faces, 3)
    dist_inside = torch.norm(points - nearest_inside, dim=2)  # (num_points, num_faces)

    # Compute nearest points for outside projections (closest edge)
    # Edge 0: v0 to v1
    line_vec0 = edge1.unsqueeze(0).repeat(
        num_points, 1, 1
    )  # (num_points, num_faces, 3)
    point_vec0 = v  # (num_points, num_faces, 3)
    line_length0 = torch.norm(line_vec0, dim=2)  # (num_points, num_faces)
    dot0 = (point_vec0 * line_vec0).sum(dim=2)  # (num_points, num_faces)
    t0 = torch.clamp(dot0 / (line_length0**2 + 1e-10), 0, 1).unsqueeze(
        -1
    )  # (num_points, num_faces, 1)
    nearest0 = (
        triangle_vertices[:, 0, :].unsqueeze(0) + line_vec0 * t0
    )  # (num_points, num_faces, 3)
    dist0 = torch.norm(points - nearest0, dim=2)  # (num_points, num_faces)

    # Edge 1: v1 to v2
    line_vec1 = (
        (triangle_vertices[:, 2, :] - triangle_vertices[:, 1, :])
        .unsqueeze(0)
        .repeat(num_points, 1, 1)
    )
    point_vec1 = points - triangle_vertices[:, 1, :].unsqueeze(0)
    line_length1 = torch.norm(line_vec1, dim=2)
    dot1 = (point_vec1 * line_vec1).sum(dim=2)
    t1 = torch.clamp(dot1 / (line_length1**2 + 1e-10), 0, 1).unsqueeze(-1)
    nearest1 = triangle_vertices[:, 1, :].unsqueeze(0) + line_vec1 * t1
    dist1 = torch.norm(points - nearest1, dim=2)

    # Edge 2: v2 to v0
    line_vec2 = (
        (triangle_vertices[:, 0, :] - triangle_vertices[:, 2, :])
        .unsqueeze(0)
        .repeat(num_points, 1, 1)
    )
    point_vec2 = points - triangle_vertices[:, 2, :].unsqueeze(0)
    line_length2 = torch.norm(line_vec2, dim=2)
    dot2 = (point_vec2 * line_vec2).sum(dim=2)
    t2 = torch.clamp(dot2 / (line_length2**2 + 1e-10), 0, 1).unsqueeze(-1)
    nearest2 = triangle_vertices[:, 2, :].unsqueeze(0) + line_vec2 * t2
    dist2 = torch.norm(points - nearest2, dim=2)

    # Stack distances and find the minimum for outside projections
    edge_dists = torch.stack([dist0, dist1, dist2], dim=2)  # (num_points, num_faces, 3)
    min_edge_dists, min_edge_indices = torch.min(
        edge_dists, dim=2
    )  # (num_points, num_faces)

    # Choose the minimum distance between inside and outside projections
    dist_outside = min_edge_dists  # (num_points, num_faces)
    dist_all = torch.where(
        inside_mask, dist_inside, dist_outside
    )  # (num_points, num_faces)

    # Find the minimum distance and corresponding face index for each point
    min_distances, min_face_indices = torch.min(
        dist_all, dim=1
    )  # (num_points,), (num_points,)

    # Gather the nearest points based on the minimum face indices
    # nearest_face_indices_expanded = min_face_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3)  # (num_points, 1, 3)
    # nearest_points = torch.gather(nearest_inside, 1, nearest_face_indices_expanded).squeeze(1)  # (num_points, 3)

    # Initialize barycentric coordinates
    barycentric = torch.zeros((num_points, 3), device=device)

    # Gather min_face specific u, v, w
    batch_indices = torch.arange(num_points, device=device)
    u_selected = u_coord[batch_indices, min_face_indices]
    v_selected = v_coord[batch_indices, min_face_indices]
    w_selected = w_coord[batch_indices, min_face_indices]
    inside_selected = inside_mask[batch_indices, min_face_indices]

    # Assign barycentric coordinates for points with inside projections
    barycentric[inside_selected] = torch.stack(
        [
            u_selected[inside_selected],
            v_selected[inside_selected],
            w_selected[inside_selected],
        ],
        dim=1,
    )

    # Handle outside projections
    outside_selected = ~inside_selected
    if outside_selected.any():
        # Get corresponding edge indices for outside points
        closest_edges = min_edge_indices[
            outside_selected, min_face_indices[outside_selected]
        ]

        # Get t values for the closest edges
        t0_selected = t0[outside_selected, min_face_indices[outside_selected], 0]
        t1_selected = t1[outside_selected, min_face_indices[outside_selected], 0]
        t2_selected = t2[outside_selected, min_face_indices[outside_selected], 0]

        # Initialize barycentric coordinates for outside points
        barycentric[outside_selected] = torch.where(
            closest_edges.unsqueeze(-1) == 0,
            torch.stack(
                [1 - t0_selected, t0_selected, torch.zeros_like(t0_selected)], dim=1
            ),
            torch.where(
                closest_edges.unsqueeze(-1) == 1,
                torch.stack(
                    [torch.zeros_like(t1_selected), 1 - t1_selected, t1_selected], dim=1
                ),
                torch.stack(
                    [t2_selected, torch.zeros_like(t2_selected), 1 - t2_selected], dim=1
                ),
            ),
        )

    # Get nearest face vertex indices
    # nearest_face_indices_np = min_face_indices.cpu().detach().numpy()
    # nearest_barycentric_np = barycentric.cpu().detach().numpy()
    # nearest_face_vertex_indices = submesh_faces[nearest_face_indices_np].cpu().detach().numpy()

    return min_distances, submesh_faces[min_face_indices], barycentric


def extract_submesh(vertices, faces, vertex_indices):
    """
    Extract submesh containing specified vertices and connected faces.

    Parameters:
    - vertices (np.ndarray, shape=(num_vertices, 3)): Full mesh vertices.
    - faces (np.ndarray, shape=(num_faces, 3)): Full mesh faces.
    - vertex_indices (list): List of vertex indices to include.

    Returns:
    - np.ndarray, shape=(num_submesh_vertices, 3): Vertices of submesh.
    - np.ndarray, shape=(num_submesh_faces, 3): Faces of submesh.
    """
    # Create mask for faces that use any of the specified vertices
    in_submesh_mask = np.isin(faces, vertex_indices)
    submesh_face_mask = np.all(in_submesh_mask, axis=1)
    submesh_faces = faces[submesh_face_mask]

    # Get unique vertices used in submesh faces
    unique_vertices = np.unique(submesh_faces)

    # Create vertex mapping
    vertex_map = np.full(len(vertices), -1)
    vertex_map[unique_vertices] = np.arange(len(unique_vertices))

    # Remap face indices
    submesh_faces = vertex_map[submesh_faces]

    # Extract vertices
    submesh_vertices = vertices[unique_vertices]

    return submesh_vertices, submesh_faces


def test_calculate_nearest_point():
    # Initialize rerun
    rr.init("calculate_nearest_point_test", spawn=True)
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Define more target points in a grid pattern
    x, y, z = np.meshgrid(
        np.linspace(-0.5, 2.0, 3), np.linspace(-0.5, 2.0, 3), np.linspace(-0.5, 2.0, 3)
    )
    target_points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    # Define vertices of a simple triangular mesh
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, -1.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    # Define faces of the mesh (each face is a triangle)
    submesh_faces = np.array([[0, 1, 2], [0, 1, 3]])

    # Calculate face normals
    face_normals = compute_vertex_normals(vertices.numpy(), submesh_faces)

    # Call the calculate_nearest_point function
    print(f"target_points.shape: {target_points.shape}")
    print(f"vertices.shape: {vertices.shape}")
    print(f"submesh_faces.shape: {submesh_faces.shape}")
    nearest_distance, nearest_face_indices, nearest_barycentric = (
        calculate_nearest_point(target_points, vertices, submesh_faces)
    )

    # Log the target points
    rr.log(
        "target_points",
        rr.Points3D(
            positions=target_points,
            radii=0.01,
            colors=np.full((len(target_points), 3), [255, 0, 0], dtype=np.uint8),
        ),
    )

    # Log the mesh
    rr.log(
        "mesh",
        rr.Mesh3D(
            vertex_positions=vertices.numpy(),
            triangle_indices=submesh_faces,
            vertex_normals=face_normals,
            vertex_colors=np.full((vertices.shape[0], 3), [0, 255, 0], dtype=np.uint8),
        ),
    )

    # Calculate and log the nearest points
    face_vertices = vertices[nearest_face_indices].numpy()
    nearest_points = np.einsum("ij,ijk->ik", nearest_barycentric, face_vertices)

    # Log nearest points
    rr.log(
        "nearest_points",
        rr.Points3D(
            positions=nearest_points,
            radii=0.01,
            colors=np.full((len(nearest_points), 3), [0, 0, 255], dtype=np.uint8),
        ),
    )

    # Log lines connecting target points to nearest points
    for i in range(target_points.shape[0]):
        rr.log(
            f"line/line_{i}",
            rr.LineStrips3D(
                strips=[[target_points[i], nearest_points[i]]],
                colors=[255, 255, 0],
                labels=[
                    f"target_point_{target_points[i]}-nearest_point_{nearest_points[i]}-distance_{nearest_distance[i]}"
                ],
            ),
        )

    # Log face normals at face centroids
    face_centroids = np.mean(vertices[submesh_faces].numpy(), axis=1)
    face_normals = compute_vertex_normals(vertices.numpy(), submesh_faces)
    normal_scale = 0.2  # Scale factor for normal visualization

    rr.log(
        "face_normals",
        rr.Arrows3D(
            origins=face_centroids,
            vectors=face_normals * normal_scale,
            radii=0.005,
            colors=np.full((len(face_centroids), 3), [255, 128, 0], dtype=np.uint8),
        ),
    )

    # Print some statistics
    print(f"Total points: {len(target_points)}")
    print(f"Face assignment counts: {np.bincount(nearest_face_indices.reshape(-1))}")


if __name__ == "__main__":
    test_calculate_nearest_point()
