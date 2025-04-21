import torch
from jaxtyping import Float
from torch import Tensor, cat, matmul


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional points into (n+1)-dimensional homogeneous points."""
    ones = torch.ones_like(points[..., :1])
    return cat([points, ones], dim=-1)


def homogenize_vectors(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional vectors into (n+1)-dimensional homogeneous vectors."""
    zeros = torch.zeros_like(points[..., :1])
    return cat([points, zeros], dim=-1)


def transform_rigid(
    xyz: Float[Tensor, "*#batch 4"],
    transform: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Apply a rigid-body transform to homogeneous points or vectors."""
    return matmul(transform, xyz.unsqueeze(-1)).squeeze(-1)


def transform_world2cam(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D world coordinates to homogeneous
    3D camera coordinates.
    """
    world2cam = torch.linalg.inv(cam2world)
    return transform_rigid(xyz, world2cam)


def transform_cam2world(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D camera coordinates to homogeneous
    3D world coordinates.
    """
    return transform_rigid(xyz, cam2world)


def project(
    xyz: Float[Tensor, "batch vertex 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
) -> Float[Tensor, "batch vertex 2"]:
    """Project homogenized 3D points in camera coordinates to pixel coordinates."""
    xyz_proj = torch.matmul(intrinsics[:, None, :, :], xyz[..., :3].unsqueeze(-1)).squeeze(-1)
    xy = xyz_proj[..., :2] / xyz_proj[..., 2:3]
    # print(f"xy: {xy}")
    return xy

