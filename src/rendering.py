import torch
from torch import Tensor
from jaxtyping import Float
from src.geometry import homogenize_points, transform_world2cam, project


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
    num_vertices = vertices.shape[0]

    # Step 1: Homogenize points [V, 3] → [V, 4]
    homo_vertices = homogenize_points(vertices)  # [V, 4]

    # Step 2: Expand points to batch [B, V, 4]
    homo_vertices = homo_vertices.unsqueeze(0).expand(batch_size, -1, -1)  # [B, V, 4]

    # Step 3: Transform to camera space
    cam_coords = transform_world2cam(homo_vertices, extrinsics)  # [B, V, 4]

    # Step 4: Project to image plane
    pixel_coords = project(cam_coords, intrinsics)  # [B, V, 2]

    # Step 5: Convert to integer pixel coordinates
    pixel_coords_rounded = pixel_coords.round().long()  # [B, V, 2]

    # Step 6: Clamp to image boundaries
    x = pixel_coords_rounded[..., 0].clamp(0, width - 1)
    y = pixel_coords_rounded[..., 1].clamp(0, height - 1)

    # Step 7: Render canvas (1.0 = white, 0.0 = black)
    canvas = torch.ones((batch_size, height, width), dtype=torch.float32)

    for b in range(batch_size):
        canvas[b, y[b], x[b]] = 0.0

    return canvas