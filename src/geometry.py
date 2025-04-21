from jaxtyping import Float
from torch import Tensor
import torch


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional points into (n+1)-dimensional homogeneous points."""
    print("homogenize_points ok")
    ones = torch.ones_like(points[..., :1])
    return torch.cat([points, ones], dim=-1)


def homogenize_vectors(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional vectors into (n+1)-dimensional homogeneous vectors."""
    print("homogenize_vectors ok")
    zeros = torch.zeros_like(points[..., :1])
    return torch.cat([points, zeros], dim=-1)


def transform_rigid(
    xyz: Float[Tensor, "*#batch 4"],
    transform: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Apply a rigid-body transform to homogeneous points or vectors."""
    print("transform_rigid ok")
    return torch.matmul(transform, xyz.unsqueeze(-1)).squeeze(-1)


def transform_world2cam(
    xyz,#: Float[Tensor, "*#batch 4"],
    cam2world#: Float[Tensor, "*#batch 4 4"],
):# -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D world coordinates to homogeneous 3D camera coordinates."""
    print("transform_world2cam ok")
    world2cam = torch.linalg.inv(cam2world)
    return transform_rigid(xyz, world2cam)


def transform_cam2world(
    xyz,# Float[Tensor, "*#batch 4"],
    cam2world#: Float[Tensor, "*#batch 4 4"],
):# -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D camera coordinates to homogeneous 3D world coordinates."""
    print("transform_cam2world ok")
    return transform_rigid(xyz, cam2world)


def project(
    xyz,#: Float[Tensor, "*#batch 4"],
    intrinsics#: Float[Tensor, "*#batch 3 3"],
):# -> Float[Tensor, "*batch 2"]:
    """Project homogenized 3D points in camera coordinates to pixel coordinates."""
    print("project ok")
    xyz_cam = xyz[..., :3]
    xy = xyz_cam[..., :2]
    z = xyz_cam[..., 2:3]
    normalized_xy = xy / z
    normalized_homo = torch.cat([normalized_xy, torch.ones_like(z)], dim=-1)
    pixel_coords = torch.matmul(intrinsics, normalized_homo.unsqueeze(-1)).squeeze(-1)
    return pixel_coords[..., :2]
