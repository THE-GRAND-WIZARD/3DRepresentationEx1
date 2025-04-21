import torch
from torch import Tensor
from jaxtyping import Float

from src.geometry import homogenize_points, project, transform_world2cam

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
    height, width = resolution
    canvas = torch.ones((batch_size, height, width), dtype=torch.float32)

    verts_h = homogenize_points(vertices)  # [vertex, 4]
    verts_h = verts_h.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, vertex, 4]

    verts_h_flat = verts_h.reshape(-1, 4)  # [batch*vertex, 4]
    extrinsics_flat = extrinsics.repeat_interleave(verts_h.shape[1], dim=0)  # [batch*vertex, 4, 4]

    cam_coords = transform_world2cam(verts_h_flat, extrinsics_flat)  # [batch*vertex, 4]
    cam_coords = cam_coords.reshape(batch_size, -1, 4)  # [batch, vertex, 4]

    pixel_coords = project(cam_coords, intrinsics)  # [batch, vertex, 2]
    pixel_coords = (pixel_coords * 256).round().long()  # [batch, vertex, 2]

    x, y = pixel_coords.unbind(-1)  # [batch, vertex]
    valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    for b in range(batch_size):
        xb = x[b][valid_mask[b]]
        yb = y[b][valid_mask[b]]
        canvas[b, yb, xb] = 0.0

    return canvas
