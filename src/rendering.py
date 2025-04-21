import torch
from torch import Tensor
from jaxtyping import Float

def render_point_cloud(
    vertices: Float[Tensor, "vertex 3"],
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    resolution: tuple[int, int] = (256, 256),
) -> Float[Tensor, "batch height width"]:
    """Create a white canvas with the specified resolution. Then, transform the points
    into camera space, project them onto the image plane, and color the corresponding
    pixels on the canvas black.
    """
    batch_size = extrinsics.shape[0]
    num_vertices = vertices.shape[0]
    height, width = resolution

    canvas = torch.ones((batch_size, height, width), dtype=torch.float32)

    vertices_homo = torch.cat([vertices, torch.ones(num_vertices, 1)], dim=1)
    vertices_homo = vertices_homo.T

    cam_coords = torch.matmul(extrinsics, vertices_homo[None, :, :])
    cam_coords = cam_coords[:, :3, :]

    z = cam_coords[:, 2, :] + 1e-5
    normalized = cam_coords[:, :2, :] / z.unsqueeze(1)
    ones = torch.ones((batch_size, 1, num_vertices), device=vertices.device)
    proj_points = torch.cat([normalized, ones], dim=1)

    pixels = torch.matmul(intrinsics, proj_points)
    u = pixels[:, 0, :].long()
    v = pixels[:, 1, :].long()

    for b in range(batch_size):
        mask = (
            (u[b] >= 0) & (u[b] < width) &
            (v[b] >= 0) & (v[b] < height) &
            (z[b] > 0)
        )
        canvas[b, v[b, mask], u[b, mask]] = 0.0

    return canvas
