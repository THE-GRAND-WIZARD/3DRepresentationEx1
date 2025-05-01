import os
import sys
sys.path.append(os.getcwd())
import env
import src.utils.engine as engine
import src.utils.utils as utils
from src.image_rectification import find_matches, show_matches
from pathlib import Path
from typing import List, Tuple

from src.calibrate_camera import (
    calibrate_camera,
    find_chessboard_corners,
    refine_corners,
    get_3D_object_points,
    load_grayscale_image,
)

import cv2
import numpy as np
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image


def recover_fundamental_matrix(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    good_matches: List[cv2.DMatch]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Recover the fundamental matrix from the good matches
    """
    pts1, pts2 = parse_matches(kp1, kp2, good_matches)
    F, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.99,
    )
    return F, mask, pts1, pts2


def compute_essential_matrix(
    camera_matrix: np.ndarray,
    fundamental_matrix: np.ndarray
) -> np.ndarray:
    """Computes the essential matrix E = K^T F K"""
    return camera_matrix.T @ fundamental_matrix @ camera_matrix


def estimate_initial_RT(
    E: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Returns two candidate rotations and translations from E"""
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    if np.linalg.det(R1) < 0:
        R1 *= -1
    if np.linalg.det(R2) < 0:
        R2 *= -1
    t1 = U[:, 2].reshape(3, 1)
    t2 = -t1
    return [R1, R2], [t1, t2]


def find_best_RT(
    candidate_Rs: List[np.ndarray],
    candidate_ts: List[np.ndarray],
    inlier_pts1: np.ndarray,
    inlier_pts2: np.ndarray,
    camera_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Chooses R, t producing max positive-depth points"""
    best_count = 0
    best_R, best_t = None, None
    P1 = get_identity_projection_matrix(camera_matrix)
    for R in candidate_Rs:
        for t in candidate_ts:
            P2 = get_local_projection_matrix(camera_matrix, R, t)
            pts4d = cv2.triangulatePoints(P1, P2, inlier_pts1, inlier_pts2)
            pts3d = (pts4d[:3] / pts4d[3]).T
            depth1 = pts3d[:, 2] > 0
            pts_cam2 = (R @ pts3d.T + t).T
            depth2 = pts_cam2[:, 2] > 0
            count = np.sum(depth1 & depth2)
            if count > best_count:
                best_count = count
                best_R, best_t = R, t
    return best_R, best_t


def get_identity_projection_matrix(camera_matrix: np.ndarray) -> np.ndarray:
    """K [I|0]"""
    return camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))


def get_local_projection_matrix(
    camera_matrix: np.ndarray,
    R: np.ndarray,
    T: np.ndarray
) -> np.ndarray:
    """K [R|T]"""
    return camera_matrix @ np.hstack((R, T))


def calibrate_camera_from_chessboard(
    image_path: Path,
    chessboard_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs camera calibration given a chessboard image"""
    image = utils.load_image(image_path)
    gray = load_grayscale_image(image)
    corners = find_chessboard_corners(gray, chessboard_size)
    corners = refine_corners(gray, corners)
    obj_pts = get_3D_object_points(chessboard_size)
    K, dist = calibrate_camera(obj_pts, corners, gray.shape[::-1])
    return K, dist


def undistort_images(
    folder: Path,
    out_folder: Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray
) -> np.ndarray:
    """Undistort all images in `folder` using cv2.undistort"""
    if out_folder.exists():
        shutil.rmtree(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    for fn in tqdm(os.listdir(folder), desc='Undistorting'):
        img = utils.load(folder / fn)
        corr = cv2.undistort(img, camera_matrix, dist_coeffs, None, camera_matrix)
        Image.fromarray(corr).save(out_folder / fn)
    h, w = img.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    return new_K


def parse_matches(
    keypoints1: List[cv2.KeyPoint],
    keypoints2: List[cv2.KeyPoint],
    good_matches: List[cv2.DMatch]
) -> Tuple[np.ndarray, np.ndarray]:
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
    return pts1, pts2


def get_inliers(
    mask: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    in1 = pts1[mask.ravel() == 1]
    in2 = pts2[mask.ravel() == 1]
    return in1.T, in2.T


def show_points_matplotlib(points3D: np.ndarray) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], s=5)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.show()


if __name__ == '__main__':
    if not os.path.exists(env.p7.output):
        os.makedirs(env.p7.output)
    expected_R = np.load(env.p7.expected_R)
    expected_T = np.load(env.p7.expected_T)

    parser = __import__('argparse').ArgumentParser()
    parser.add_argument('--setup', action='store_true')
    setup = parser.parse_args().setup
    chessboard_size = (16, 10)
    if setup:
        engine.get_chessboard(env.p7.chessboard)
        engine.get_object_images(
            env.p7.arc_obj,
            env.p7.arc_texture,
            env.p7.raw_images,
            views=5,
        )

    camera_matrix, dist = calibrate_camera_from_chessboard(
        env.p7.chessboard,
        chessboard_size,
    )

    im1 = utils.load_image(env.p7.raw_images / 'object_0.png')
    im2 = utils.load_image(env.p7.raw_images / 'object_1.png')

    kp1, kp2, good = find_matches(im1, im2)
    show_matches(im1, im2, kp1, kp2, good)

    F, mask, pts1, pts2 = recover_fundamental_matrix(kp1, kp2, good)
    in1, in2 = get_inliers(mask, pts1, pts2)
    E = compute_essential_matrix(camera_matrix, F)
    Rs, Ts = estimate_initial_RT(E)
    R, T = find_best_RT(Rs, Ts, in1, in2, camera_matrix)
    # assert np.allclose(R, expected_R, atol=1e-2)
    # assert np.allclose(T, expected_T, atol=1e-2)
    np.save(env.p7.rotation_matrix, R)
    np.save(env.p7.translation_matrix, T)

    P1 = get_identity_projection_matrix(camera_matrix)
    P2 = get_local_projection_matrix(camera_matrix, R, T)
    pts4D = cv2.triangulatePoints(P1, P2, in1, in2)
    pts3D = (pts4D[:3] / pts4D[3]).T
    print(f"Triangulated {len(pts3D)} points.")
    print("Singular values of F:", np.linalg.svd(F)[1])
    show_points_matplotlib(pts3D)
    np.save(env.p7.pointcloud, pts3D)
