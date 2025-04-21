from tkinter import Image

import numpy as np
import os
import sys

from PIL import ImageDraw

import env
import src.utils.utils as utils
import matplotlib.pyplot as plt

def lstsq_eight_point_alg(points1: np.array, points2: np.array) -> np.array:
    """
    Least Squares Eight-Point Algorithm for computing the fundamental matrix.
    """
    assert points1.shape == points2.shape
    n = points1.shape[0]

    # Construct matrix W
    W = np.zeros((n, 9))
    for i in range(n):
        x1, y1, _ = points1[i]
        x2, y2, _ = points2[i]
        W[i] = [x1 * x2, x1 * y2, x1,
                y1 * x2, y1 * y2, y1,
                x2, y2, 1]

    # Solve Wf = 0 using SVD
    U, S, Vt = np.linalg.svd(W)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint
    Uf, Sf, Vtf = np.linalg.svd(F)
    Sf[-1] = 0
    F_rank2 = Uf @ np.diag(Sf) @ Vtf

    return F_rank2


def normalize_points(points: np.array) -> tuple:
    """
    Normalize a set of homogeneous points: centroid at origin, average distance sqrt(2).
    Returns normalized points and transformation matrix.
    """
    points = points / points[:, 2][:, np.newaxis]
    centroid = np.mean(points[:, :2], axis=0)
    shifted = points[:, :2] - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
    scale = np.sqrt(2) / mean_dist

    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0,     0,                 1]
    ])
    normalized_points = (T @ points.T).T

    return normalized_points, T


def normalized_eight_point_alg(points1: np.array, points2: np.array) -> np.array:
    """
    Normalized Eight-Point Algorithm for computing the fundamental matrix.
    """
    norm_p1, T1 = normalize_points(points1)
    norm_p2, T2 = normalize_points(points2)

    F_norm = lstsq_eight_point_alg(norm_p1, norm_p2)

    # Denormalize
    F_denorm = T2.T @ F_norm @ T1
    return F_denorm


def compute_epipolar_lines(points: np.array, F: np.array) -> np.array:
    """
    Compute epipolar lines from point correspondences and a fundamental matrix.
    """
    lines = (F @ points.T).T  # shape (N, 3), lines in form Ax + By + C = 0
    line_params = []
    for A, B, C in lines:
        if B == 0:
            m, b = 1e6, 1e6  # vertical line, approximate with large slope
        else:
            m = -A / B
            b = -C / B
        line_params.append((m, b))
    return np.array(line_params)


def show_epipolar_imgs(img1: np.ndarray,
                       img2: np.ndarray,
                       lines1: np.ndarray,
                       lines2: np.ndarray,
                       pts1: np.ndarray,
                       pts2: np.ndarray,
                       offset: int = 0) -> np.ndarray:
    epi_img1 = get_epipolar_img(img1, lines1, pts1)
    epi_img2 = get_epipolar_img(img2, lines2, pts2)

    if offset < 0:
        h1, w1, c1 = epi_img1.shape
        padding = np.zeros((-offset, w1, c1), dtype=epi_img1.dtype)
        epi_img1 = np.vstack((padding, epi_img1))
    else:
        h2, w2, c2 = epi_img2.shape
        padding = np.zeros((offset, w2, c2), dtype=epi_img1.dtype)
        epi_img2 = np.vstack((padding, epi_img2))

    h1, w1, c1 = epi_img1.shape
    h2, w2, c2 = epi_img2.shape

    max_h = max(h1, h2)

    if h1 < max_h:
        pad_height = max_h - h1
        padding = np.zeros((pad_height, w1, c1), dtype=epi_img1.dtype)
        epi_img1 = np.vstack((padding, epi_img1))

    if h2 < max_h:
        pad_height = max_h - h2
        padding = np.zeros((pad_height, w2, c2), dtype=epi_img2.dtype)
        epi_img2 = np.vstack((epi_img2, padding))

    combined_img = np.hstack((epi_img1, epi_img2))
    plt.imshow(combined_img)
    plt.title("Epipolar Lines")
    plt.show()

    return combined_img


def draw_points(img: np.ndarray,
                points: np.ndarray,
                color: tuple = (0, 255, 0),
                radius: int = 5) -> np.ndarray:
    img_with_corners = Image.fromarray(img)
    draw = ImageDraw.Draw(img_with_corners)

    for (x, y, _) in points:
        left_up_point = (x - radius, y - radius)
        right_down_point = (x + radius, y + radius)
        draw.ellipse([left_up_point, right_down_point], outline=color, width=2)

    return np.array(img_with_corners)


def draw_lines(img: np.ndarray,
               lines: np.ndarray,
               color: tuple = (255, 0, 0),
               thickness: int = 3) -> np.ndarray:
    from PIL import Image, ImageDraw
    import numpy as np

    img_with_lines = Image.fromarray(img)
    draw = ImageDraw.Draw(img_with_lines)
    width, _ = img_with_lines.size

    for (m, b) in lines:
        # Compute two endpoints using x = 0 and x = width.
        x1 = 0
        y1 = m * x1 + b
        x2 = width
        y2 = m * x2 + b

        draw.line([(x1, y1), (x2, y2)], fill=color, width=thickness)

    return np.array(img_with_lines)


def compute_distance_to_epipolar_lines(points1: np.array,
                                       points2: np.array,
                                       F: np.array) -> float:
    l = F.T.dot(points2.T)
    # distance from point(x0, y0) to line: Ax + By + C = 0 is
    # |Ax0 + By0 + C| / sqrt(A^2 + B^2)
    d = np.mean(np.abs(np.sum(l * points1.T, axis=0)) / np.sqrt(l[0, :] ** 2 + l[1, :] ** 2))
    return d


def get_epipolar_img(img: np.ndarray,
                     lines: np.ndarray,
                     points: np.ndarray) -> np.ndarray:
    lines_img = draw_lines(img, lines)
    points_img = draw_points(lines_img, points)
    return points_img


if __name__ == '__main__':
    if not os.path.exists(env.p5.output):
        os.makedirs(env.p5.output)
    expected_F_LLS = np.load(env.p5.expected_F_LLS)
    expected_dist_im1_LLS, expected_dist_im2_LLS = np.load(env.p5.expected_dist_LLS)

    expected_F_normalized = np.load(env.p5.expected_F_normalized)
    expected_dist_im1_normalized, expected_dist_im2_normalized = np.load(env.p5.expected_dist_normalized)

    im1 = utils.load_image(env.p5.const_im1)
    im2 = utils.load_image(env.p5.const_im2)

    points1 = utils.load_points(env.p5.pts_1)
    points2 = utils.load_points(env.p5.pts_2)
    assert (points1.shape == points2.shape)

    # Part 5.a
    F_lls = lstsq_eight_point_alg(points1, points2)
    print("Fundamental Matrix from LLS  8-point algorithm:\n", F_lls)
    assert np.allclose(F_lls, expected_F_LLS,
                       atol=1e-2), f"Fundamental matrix does not match this expected matrix:\n{expected_F_LLS}"
    np.save(env.p5.F_LLS, F_lls)

    dist_im1_LLS = compute_distance_to_epipolar_lines(points1, points2, F_lls)
    dist_im2_LLS = compute_distance_to_epipolar_lines(points2, points1, F_lls.T)
    print("Distance to lines in image 1 for LLS:", \
          dist_im1_LLS)
    print("Distance to lines in image 2 for LLS:", \
          dist_im2_LLS)
    assert np.allclose(dist_im1_LLS, expected_dist_im1_LLS,
                       atol=1e-2), f"Distance to lines in image 1 does not match this expected distance: {expected_dist_im1_LLS}"
    assert np.allclose(dist_im2_LLS, expected_dist_im2_LLS,
                       atol=1e-2), f"Distance to lines in image 2 does not match this expected distance: {expected_dist_im2_LLS}"
    np.save(env.p5.dist_LLS, np.array([dist_im1_LLS, dist_im2_LLS]))

    # Part 5.b
    F_normalized = normalized_eight_point_alg(points1, points2)
    print("Fundamental Matrix from normalized 8-point algorithm:\n", \
          F_normalized)
    assert np.allclose(F_normalized, expected_F_normalized,
                       atol=1e-2), f"Fundamental matrix does not match this expected matrix:\n{expected_F_normalized}"

    dist_im1_normalized = compute_distance_to_epipolar_lines(points1, points2, F_normalized)
    dist_im2_normalized = compute_distance_to_epipolar_lines(points2, points1, F_normalized.T)
    print("Distance to lines in image 1 for normalized:", \
          dist_im1_normalized)
    print("Distance to lines in image 2 for normalized:", \
          dist_im2_normalized)
    assert np.allclose(dist_im1_normalized, expected_dist_im1_normalized,
                       atol=1e-2), f"Distance to lines in image 1 does not match this expected distance: {expected_dist_im1_normalized}"
    assert np.allclose(dist_im2_normalized, expected_dist_im2_normalized,
                       atol=1e-2), f"Distance to lines in image 2 does not match this expected distance: {expected_dist_im2_normalized}"
    np.save(env.p5.dist_normalized, np.array([dist_im1_normalized, dist_im2_normalized]))

    # Part 5.c
    lines1 = compute_epipolar_lines(points2, F_lls.T)
    lines2 = compute_epipolar_lines(points1, F_lls)
    lls_img = show_epipolar_imgs(im1, im2, lines1, lines2, points1, points2)
    Image.fromarray(lls_img).save(env.p5.lls_img)

    lines1 = compute_epipolar_lines(points2, F_normalized.T)
    lines2 = compute_epipolar_lines(points1, F_normalized)
    norm_img = show_epipolar_imgs(im1, im2, lines1, lines2, points1, points2)
    Image.fromarray(norm_img).save(env.p5.norm_img)