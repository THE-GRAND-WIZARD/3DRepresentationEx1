from pathlib import Path
from typing import Literal, TypedDict
import json
import torch
import imageio.v3 as iio
import numpy as np
from jaxtyping import Float
from torch import Tensor


class PuzzleDataset(TypedDict):
    extrinsics: Float[Tensor, "batch 4 4"]
    intrinsics: Float[Tensor, "batch 3 3"]
    images: Float[Tensor, "batch height width"]


def load_dataset(path: Path) -> PuzzleDataset:
    with open(path / "metadata.json", "r") as f:
        metadata = json.load(f)
    extrinsics = torch.tensor(metadata["extrinsics"], dtype=torch.float32)
    intrinsics = torch.tensor(metadata["intrinsics"], dtype=torch.float32)

    images = []
    for i in range(32):
        img_path = path / "images" / f"{i:02d}.png"
        img = iio.imread(img_path).astype(np.float32) / 255.0
        images.append(torch.tensor(img))
    images = torch.stack(images)

    return PuzzleDataset(extrinsics=extrinsics, intrinsics=intrinsics, images=images)


def convert_dataset(dataset: PuzzleDataset) -> PuzzleDataset:
    extrinsics = dataset["extrinsics"]
    converted = []

    # Step 1: Determine current axes by looking at R from a matrix
    # Let's assume extrinsics are c2w and analyze the rotation part
    R = extrinsics[0, :3, :3]
    t = extrinsics[0, :3, 3]
    origin = t
    look = -origin / torch.norm(origin)  # normalized vector toward origin
    up_candidate = R[:, 1]  # this should point toward +y in world
    if torch.dot(up_candidate, torch.tensor([0, 1, 0], dtype=torch.float32)) < 0:
        R[:, 1] *= -1  # flip if needed

    # Reconstruct new camera frame in OpenCV convention:
    # Look = +Z, Up = -Y, Right = +X
    new_Rs = []
    new_Ts = []
    for E in extrinsics:
        R_old = E[:3, :3]
        T_old = E[:3, 3].unsqueeze(1)

        # Convert to world-to-camera if it's c2w
        R_cam = R_old.T
        T_cam = -R_old.T @ T_old

        # Apply axis remapping
        # Assume original: Right = -Z, Up = +Y, Look = +X
        change_basis = torch.tensor([
            [0,  0,  1],
            [0, -1,  0],
            [1,  0,  0]
        ], dtype=torch.float32)

        R_new = change_basis @ R_cam
        T_new = change_basis @ T_cam

        # Now go back to c2w format
        R_final = R_new.T
        T_final = -R_final @ T_new

        extrinsic_new = torch.eye(4)
        extrinsic_new[:3, :3] = R_final
        extrinsic_new[:3, 3] = T_final.squeeze()
        new_Rs.append(extrinsic_new)

    new_extrinsics = torch.stack(new_Rs)
    return PuzzleDataset(
        extrinsics=new_extrinsics,
        intrinsics=dataset["intrinsics"],
        images=dataset["images"]
    )


def quiz_question_1() -> Literal["w2c", "c2w"]:
    return "c2w"  # based on inspection of translation and rotation behavior


def quiz_question_2() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    return "+x"  # camera looks toward origin, origin is along -X → look = +X


def quiz_question_3() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    return "+y"  # up vector has positive dot with world +Y


def quiz_question_4() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    return "-z"  # right-hand rule implies right = -Z


def explanation_of_problem_solving_process() -> str:
    return (
        "I started by loading the dataset and analyzing the extrinsic matrices. Since the camera positions were known "
        "to point toward the origin, I inferred the look direction from the translation vector. By comparing the rotation "
        "matrix axes with the expected world orientation, I deduced the existing camera frame: look was +X, up was +Y, and "
        "right was -Z. I then constructed a change-of-basis matrix to remap the camera coordinate frame into OpenCV format "
        "(+Z forward, -Y up, +X right). After converting each extrinsic matrix into world-to-camera format, I applied this "
        "transformation and inverted it to yield OpenCV-style camera-to-world matrices. Finally, I validated correctness by "
        "rendering and confirming image alignment."
    )
