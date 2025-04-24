import os
import sys
import env
import src.utils.utils as utils
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

import numpy as np


def lstsq_eight_point_alg(points1: np.array, points2: np.array) -> np.array:
    '''
    Computes the fundamental matrix from matching points using 
    linear least squares eight point algorithm
    Arguments:
        points1 - N points in the first image that match with points2
        points2 - N points in the second image that match with points1

        Both points1 and points2 are from the get_data_from_txt_file() method
    Returns:
        F - the fundamental matrix such that (points2)^T * F * points1 = 0
    '''
    assert points1.shape == points2.shape
    N = points1.shape[0]

    # Convert points to homogeneous coordinates (3D)
    if points1.shape[1] == 2:
        points1 = np.hstack([points1, np.ones((N, 1))])
    if points2.shape[1] == 2:
        points2 = np.hstack([points2, np.ones((N, 1))])

    # Construct W matrix
    W = np.zeros((N, 9))
    for i in range(N):
        x1, y1, _ = points1[i]
        x2, y2, _ = points2[i]
        W[i] = [x2 * x1, x2 * y1, x2,
                y2 * x1, y2 * y1, y2,
                x1,      y1,     1]

    # Solve Wf = 0 using SVD
    _, _, Vt = np.linalg.svd(W)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank 2 constraint on F by zeroing out smallest singular value
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F_rank2 = U @ np.diag(S) @ Vt

    # Normalize F (optional: make F[2,2] = 1 if non-zero)
    if F_rank2[-1, -1] != 0:
        F_rank2 = F_rank2 / F_rank2[-1, -1]

    return F_rank2



def normalize_points(points):
    """
    Normalize 2D points (Nx2) to have centroid at origin and avg distance sqrt(2)
    """
    # ודא שהנקודות הן Nx2 בלבד
    if points.shape[1] > 2:
        points = points[:, :2]  # ניפטר מעמודות מיותרות

    # חישוב מרכז הכובד והסטה
    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    scale = np.sqrt(2) / np.mean(np.linalg.norm(shifted, axis=1))

    # מטריצת נרמול
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])

    # המרה להומוגניות (Nx3)
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    normalized_points = (T @ points_h.T).T

    return normalized_points, T


def normalized_eight_point_alg(points1: np.array, points2: np.array) -> np.array:
    '''
    Computes the fundamental matrix using the normalized 8-point algorithm
    '''
    # שלב 1: נרמול הנקודות
    norm_points1, T1 = normalize_points(points1)
    norm_points2, T2 = normalize_points(points2)

    # שלב 2: בניית מטריצת A
    N = points1.shape[0]
    A = np.zeros((N, 9))
    for i in range(N):
        x1, y1 = norm_points1[i, 0], norm_points1[i, 1]
        x2, y2 = norm_points2[i, 0], norm_points2[i, 1]
        A[i] = [x2*x1, x2*y1, x2,
                y2*x1, y2*y1, y2,
                x1,    y1,    1]

    # שלב 3: פתרון בעזרת SVD (Af = 0)
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # שלב 4: אכיפת דירוג 2 על F (מטריצת יסוד)
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0  # אפס את הערך הקטן ביותר
    F_rank2 = U @ np.diag(S) @ Vt

    # שלב 5: ביטול הנרמול
    F_final = T2.T @ F_rank2 @ T1

    return F_final


def compute_epipolar_lines(points: np.array, F: np.array) -> np.array:
    """
    Computes the epipolar lines in homogenous coordinates
    given matching points in two images and the fundamental matrix
    Arguments:
        points - N points in the first image that match with points2
        F - the Fundamental matrix such that (points1)^T * F * points2 = 0
    Returns:
        lines - the epipolar lines in slope‑intercept form (m, b)
    """
    # l = F p  ⇒  A x + B y + C = 0
    l = (F @ points.T).T      # shape (N,3)

    # Convert to y = m x + b  (skip nearly vertical lines gracefully)
    A, B, C = l[:, 0], l[:, 1], l[:, 2]
    epsilon = 1e-12
    m = -A / (B + epsilon)
    b = -C / (B + epsilon)
    return np.column_stack([m, b])


def show_epipolar_imgs(img1: np.ndarray, 
                       img2: np.ndarray, 
                       lines1: np.ndarray, 
                       lines2: np.ndarray, 
                       pts1: np.ndarray, 
                       pts2: np.ndarray, 
                       offset: int=0) -> np.ndarray:
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
                color: tuple=(0, 255, 0), 
                radius: int=5) -> np.ndarray:
    img_with_corners = Image.fromarray(img)
    draw = ImageDraw.Draw(img_with_corners)

    for (x, y, _) in points:
        left_up_point = (x - radius, y - radius)
        right_down_point = (x + radius, y + radius)
        draw.ellipse([left_up_point, right_down_point], outline=color, width=2)
    
    return np.array(img_with_corners)

def draw_lines(img: np.ndarray, 
               lines: np.ndarray, 
               color: tuple=(255, 0, 0), 
               thickness: int=3) -> np.ndarray:
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
    assert np.allclose(F_lls, expected_F_LLS, atol=1e-2), f"Fundamental matrix does not match this expected matrix:\n{expected_F_LLS}"
    np.save(env.p5.F_LLS, F_lls)

    dist_im1_LLS = compute_distance_to_epipolar_lines(points1, points2, F_lls)
    dist_im2_LLS = compute_distance_to_epipolar_lines(points2, points1, F_lls.T)
    print("Distance to lines in image 1 for LLS:", \
        dist_im1_LLS)
    print("Distance to lines in image 2 for LLS:", \
        dist_im2_LLS)
    assert np.allclose(dist_im1_LLS, expected_dist_im1_LLS, atol=1e-2), f"Distance to lines in image 1 does not match this expected distance: {expected_dist_im1_LLS}"
    assert np.allclose(dist_im2_LLS, expected_dist_im2_LLS, atol=1e-2), f"Distance to lines in image 2 does not match this expected distance: {expected_dist_im2_LLS}"
    np.save(env.p5.dist_LLS, np.array([dist_im1_LLS, dist_im2_LLS]))

    # Part 5.b
    F_normalized = normalized_eight_point_alg(points1, points2)
    print("Fundamental Matrix from normalized 8-point algorithm:\n", \
        F_normalized)
    assert np.allclose(F_normalized, expected_F_normalized, atol=1e-2), f"Fundamental matrix does not match this expected matrix:\n{expected_F_normalized}"

    dist_im1_normalized = compute_distance_to_epipolar_lines(points1, points2, F_normalized)
    dist_im2_normalized = compute_distance_to_epipolar_lines(points2, points1, F_normalized.T)
    print("Distance to lines in image 1 for normalized:", \
        dist_im1_normalized)
    print("Distance to lines in image 2 for normalized:", \
        dist_im2_normalized)
    assert np.allclose(dist_im1_normalized, expected_dist_im1_normalized, atol=1e-2), f"Distance to lines in image 1 does not match this expected distance: {expected_dist_im1_normalized}"
    assert np.allclose(dist_im2_normalized, expected_dist_im2_normalized, atol=1e-2), f"Distance to lines in image 2 does not match this expected distance: {expected_dist_im2_normalized}"
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
