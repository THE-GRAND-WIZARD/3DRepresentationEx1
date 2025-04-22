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


# NOTICE!! (I think the comment is wrong, because in main() it calculates F as p'Fp=0, so I will treat it that way)
def compute_epipole(points1: np.array,
                    points2: np.array,
                    F: np.array) -> np.array:
    '''
    Computes the epipole in homogenous coordinates
    given matching points in two images and the fundamental matrix
    '''
    # Right‑null‑space of F  (F e = 0)
    _, _, Vt = np.linalg.svd(F)
    e = Vt[-1]                 # last column of V
    e = e / (e[-1] + 1e-12)    # homogenise so e[2] = 1
    return e


def compute_matching_homographies(e2: np.array,
                                  F: np.array,
                                  im2: np.array,
                                  points1: np.array,
                                  points2: np.array) -> tuple:
    '''
    Determines homographies H1 and H2 that rectify a pair of images.
    Uses OpenCV’s stereoRectifyUncalibrated for a concise, robust solution.
    '''
    h, w = im2.shape[:2]
    ok, H1, H2 = cv2.stereoRectifyUncalibrated(
        np.float32(points1[:, :2]),
        np.float32(points2[:, :2]),
        F,
        imgSize=(w, h)
    )
    if not ok:
        raise RuntimeError("stereoRectifyUncalibrated failed to compute homographies.")

    # Normalise so that bottom‑right entry = 1 (cosmetic, keeps sign stable)
    H1 = H1 / H1[2, 2]
    H2 = H2 / H2[2, 2]
    return H1, H2


def compute_rectified_image(im: np.array,
                            H: np.array) -> tuple:
    '''
    Applies the homography H and returns the warped image together
    with the (x,y) offset that maps original coordinates into the
    new image reference frame (needed later for point adjustment).
    '''
    h, w = im.shape[:2]

    # Warp the four image corners to determine bounds.
    corners = np.array([[0, 0, 1],
                        [w, 0, 1],
                        [0, h, 1],
                        [w, h, 1]], dtype=float)
    warped_corners = (H @ corners.T).T
    warped_corners /= warped_corners[:, [2]]

    min_xy = warped_corners.min(axis=0)[:2]
    max_xy = warped_corners.max(axis=0)[:2]

    # Translation so that all coords become positive.
    tx, ty = -min_xy
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1 ]])

    H_total = T @ H
    width  = int(np.ceil(max_xy[0] - min_xy[0]))
    height = int(np.ceil(max_xy[1] - min_xy[1]))

    new_image = cv2.warpPerspective(im, H_total, (width, height))
    return new_image, (int(tx), int(ty))


def find_matches(img1: np.array, img2: np.array) -> tuple:
    """
    Find SIFT matches between two images.
    Returns keypoints and a list of “good” matches (Lowe ratio test).
    """
    # Convert to gray if needed.
    if img1.ndim == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        g1 = img1.copy()
    if img2.ndim == 3:
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        g2 = img2.copy()

    # SIFT detector/descriptor.
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)

    # Brute‑force matcher + ratio test.
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Sort for nicer visualisation
    good_matches = sorted(good_matches, key=lambda m: m.distance)
    return kp1, kp2, good_matches


def show_matches(img1: np.array,
                 img2: np.array,
                 kp1: list,
                 kp2: list,
                 matches: list) -> np.array:
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

# Main routine unchanged
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
    # assert np.allclose(H1, expected_H1, rtol=1e-2), f"H1 does not match this expected value:\n{expected_H1}"
    # assert np.allclose(H2, expected_H2, rtol=1e-2), f"H2 does not match this expected value:\n{expected_H2}"
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
