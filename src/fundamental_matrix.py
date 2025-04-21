import numpy as np


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