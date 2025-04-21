import os
import sys
import env
import src.utils.utils as utils

from PIL import Image
import numpy as np
import cv2
from src.fundamental_matrix import *
import matplotlib.pyplot as plt

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def compute_epipole(points1: np.array, points2: np.array, F: np.array) -> np.array:
    """
    Computes the epipole in homogeneous coordinates as the right null space of F.
    That is, F * e = 0 or F.T * e = 0.
    """
    U, S, Vt = np.linalg.svd(F)
    e = Vt[-1]
    return e / e[2]


def compute_matching_homographies(e2: np.array, F: np.array, im2: np.array,
                                  points1: np.array, points2: np.array) -> tuple:
    """
    Computes rectifying homographies for stereo image pair.
    Based on Hartley’s rectification method.
    """
    h, w = im2.shape[:2]

    # Translate the epipole to origin
    T = np.array([[1, 0, -w / 2],
                  [0, 1, -h / 2],
                  [0, 0, 1]])
    e2_h = T @ e2

    e2_x, e2_y = e2_h[0], e2_h[1]
    r = np.sqrt(e2_x**2 + e2_y**2)
    a = e2_x / r
    b = e2_y / r

    # Rotate epipole onto x-axis
    R = np.array([[a, b, 0],
                  [-b, a, 0],
                  [0, 0, 1]])
    e2_rot = R @ e2_h
    f = e2_rot[0]

    G = np.identity(3)
    G[2, 0] = -1 / f  # skew to infinity

    H2 = np.linalg.inv(T) @ np.linalg.inv(R) @ G @ R @ T
    e2x = np.array([[0, -e2[2], e2[1]],
                    [e2[2], 0, -e2[0]],
                    [-e2[1], e2[0], 0]])
    M = e2x @ F + np.outer(e2, [1, 1, 1])  # M = [e2]_x F + e2 * v^T

    # H1 = H2 * M
    H1 = H2 @ M

    return H1, H2


def compute_rectified_image(im: np.array, H: np.array) -> tuple:
    """
    Warps the input image using the given homography and returns offset.
    """
    h, w = im.shape[:2]
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1]
    ]).T
    warped_corners = H @ corners
    warped_corners = warped_corners[:2] / warped_corners[2]

    min_x = np.min(warped_corners[0])
    min_y = np.min(warped_corners[1])

    offset = np.array([-min_x, -min_y])
    offset_matrix = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ])

    H_offset = offset_matrix @ H
    new_im = cv2.warpPerspective(im, H_offset, (int(np.max(warped_corners[0] - min_x)), int(np.max(warped_corners[1] - min_y))))
    return new_im, offset


def find_matches(img1: np.array, img2: np.array) -> tuple:
    """
    Detects keypoints and computes SIFT matches between two images.
    """
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    return kp1, kp2, good


def show_matches(img1: np.array, img2: np.array, kp1: list, kp2: list, matches: list) -> np.array:
    result_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.imshow(result_img)
    plt.title("SIFT Matches")
    plt.show()
    return result_img


if __name__ == '__main__':
    if not os.path.exists(env.p6.output):
        os.makedirs(env.p6.output)
    expected_e1, expected_e2 = np.load(env.p6.expected_e1), np.load(env.p6.expected_e2)
    expected_H1, expected_H2 = np.load(env.p6.expected_H1), np.load(env.p6.expected_H2)
    im1 = utils.load_image(env.p5.const_im1)
    im2 = utils.load_image(env.p5.const_im2)

    points1 = utils.load_points(env.p5.pts_1)
    points2 = utils.load_points(env.p5.pts_2)
    assert (points1.shape == points2.shape)
    F = normalized_eight_point_alg(points1, points2)

    # Part 6.a
    e1 = compute_epipole(points1, points2, F)
    e2 = compute_epipole(points2, points1, F.transpose())
    print("e1", e1)
    print("e2", e2)
    assert np.allclose(e1, expected_e1, rtol=1e-2), f"e1 does not match this expected value:\n{expected_e1}"
    assert np.allclose(e2, expected_e2, rtol=1e-2), f"e2 does not match this expected value:\n{expected_e2}"
    np.save(env.p6.e1, e1)
    np.save(env.p6.e2, e2)

    # Part 6.b
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print("H1:\n", H1)
    print
    print("H2:\n", H2)
    assert np.allclose(H1, expected_H1, rtol=1e-2), f"H1 does not match this expected value:\n{expected_H1}"
    assert np.allclose(H2, expected_H2, rtol=1e-2), f"H2 does not match this expected value:\n{expected_H2}"
    np.save(env.p6.H1, H1)
    np.save(env.p6.H2, H2)

    # Part 6.c
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)

    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)
    total_offset_y = np.mean(new_points1[:, 1] - new_points2[:, 1]).round()

    F_new = normalized_eight_point_alg(new_points1, new_points2)
    lines1 = compute_epipolar_lines(new_points2, F_new.T)
    lines2 = compute_epipolar_lines(new_points1, F_new)
    aligned_img = show_epipolar_imgs(rectified_im1, rectified_im2, lines1, lines2, new_points1, new_points2, offset=int(total_offset_y))
    Image.fromarray(aligned_img).save(env.p6.aligned_epipolar)

    # Part 6.d
    im1 = utils.load_image(env.p5.const_im1)
    im2 = utils.load_image(env.p5.const_im2)
    kp1, kp2, good_matches = find_matches(im1, im2)
    cv_matches = show_matches(im1, im2, kp1, kp2, good_matches)
    Image.fromarray(cv_matches).save(env.p6.cv_matches)