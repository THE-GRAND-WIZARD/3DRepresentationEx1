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
    Arguments:
        points1 - N points in the first image that match with points2
        points2 - N points in the second image that match with points1
        F - the Fundamental matrix such that (points1)^T * F * points2 = 0

        Both points1 and points2 are from the get_data_from_txt_file() method
    Returns:
        epipole - the homogenous coordinates [x y 1] of the epipole in the image
    '''
        # מבצעים SVD על F
    _, _, Vt = np.linalg.svd(F)
    
    # הווקטור האחרון ב-V הוא null space של F: כלומר F * e = 0
    epipole = Vt[-1]

    # מבטיחים שהאפיפול יהיה בהומוגניות (נורמליזציה של הקואורדינטות)
    epipole = epipole / epipole[-1]

    return epipole
    



# def compute_matching_homographies(e2: np.array, 
#                                   F: np.array, 
#                                   im2: np.array, 
#                                   points1: np.array, 
#                                   points2: np.array) -> tuple:
#     '''
#     Determines homographies H1 and H2 such that they
#     rectify a pair of images
#     Arguments:
#         e2 - the second epipole (homogeneous coords)
#         F - the Fundamental matrix
#         im2 - the second image
#         points1 - N points in the first image that match with points2
#         points2 - N points in the second image that match with points1
#     Returns:
#         H1 - the homography associated with the first image
#         H2 - the homography associated with the second image
#     '''

#     h, w = im2.shape[:2]

#     # --- Step 1: Build H2 ---

#     # 1. Translate image center to origin
#     T = np.array([
#         [1, 0, -w / 2],
#         [0, 1, -h / 2],
#         [0, 0, 1]
#     ])

#     # 2. Rotate epipole to lie on x-axis
#     ex = T @ e2

#     # Compute rotation angle theta
#     theta = np.arctan2(ex[1], ex[0])

#     # Rotation matrix around origin to align epipole to x-axis
#     cos_theta = np.cos(theta)
#     sin_theta = np.sin(theta)

#     R = np.array([
#         [cos_theta, sin_theta, 0],
#         [-sin_theta, cos_theta, 0],
#         [0, 0, 1]
#     ])
#     # 3. Projective transform sending epipole to infinity
#     rotated_epipole = R @ (T @ e2)
#     f = rotated_epipole[0] / rotated_epipole[2]
#     G = np.array([
#         [1, 0, 0],
#         [0, 1, 0],
#         [-1/f, 0, 1]
#     ])

#     # Final homography H2
#     H2 = np.linalg.inv(T) @ G @ R @ T
#     # Step 5: Compute M = [e2]_x * F + e2 * v^T
#     e2_skew = np.array([
#         [0, -e2[2], e2[1]],
#         [e2[2], 0, -e2[0]],
#         [-e2[1], e2[0], 0]
#     ])
#     v = np.ones((3, 1))
#     M = e2_skew @ F + e2.reshape(3, 1) @ v.T

#     # Step 6: Transform points
#     p1_hat = (H2 @ (M @ points1.T)).T
#     p2_hat = (H2 @ points2.T).T

#     # Normalize homogeneous coordinates
#     p1_hat /= p1_hat[:, 2][:, np.newaxis]
#     p2_hat /= p2_hat[:, 2][:, np.newaxis]

#     # Step 7: Solve least squares problem for HA
#     W = np.stack([p1_hat[:, 0], p1_hat[:, 1], np.ones(len(p1_hat))], axis=1)
#     b = p2_hat[:, 0]
#     a, _, _, _ = np.linalg.lstsq(W, b, rcond=None)

#     HA = np.array([
#         [a[0], a[1], a[2]],
#         [0, 1, 0],
#         [0, 0, 1]
#     ])

#     # Step 8: Compute H1
#     H1 = HA @ H2 @ M

#     return H1, H2

import numpy as np

def compute_matching_homographies(e2: np.array, 
                                   F: np.array, 
                                   im2: np.array, 
                                   points1: np.array, 
                                   points2: np.array) -> tuple:
    h, w = im2.shape[:2]

    # Step 1: Translate second image center to origin
    T = np.array([
        [1, 0, -w / 2],
        [0, 1, -h / 2],
        [0, 0, 1]
    ])

    e2 = e2 / e2[2]
    e2_ = T @ e2

    # Step 2: Rotate e2 to lie on x-axis
    norm = np.sqrt(e2_[0]**2 + e2_[1]**2)
    alpha = 1 if e2_[0] >= 0 else -1
    R = np.array([
        [alpha * e2_[0] / norm, alpha * e2_[1] / norm, 0],
        [-alpha * e2_[1] / norm, alpha * e2_[0] / norm, 0],
        [0, 0, 1]
    ])

    e2_rotated = R @ e2_
    f = e2_rotated[0]

    # Step 3: Projective transformation to send e2 to infinity
    G = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-1 / f, 0, 1]
    ])

    # Step 4: Combine to get H2
    H2 = np.linalg.inv(T) @ G @ R @ T

    # Step 5: Compute M = [e2]_x * F + e2 * v^T
    e2_skew = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0]
    ])
    v = np.ones((3, 1))
    M = e2_skew @ F + e2.reshape(3, 1) @ v.T

    # Step 6: Transform points
    p1_hat = (H2 @ (M @ points1.T)).T
    p2_hat = (H2 @ points2.T).T

    # Normalize homogeneous coordinates
    p1_hat /= p1_hat[:, 2][:, np.newaxis]
    p2_hat /= p2_hat[:, 2][:, np.newaxis]

    # Step 7: Solve least squares problem for HA
    W = np.stack([p1_hat[:, 0], p1_hat[:, 1], np.ones(len(p1_hat))], axis=1)
    b = p2_hat[:, 0]
    a, _, _, _ = np.linalg.lstsq(W, b, rcond=None)

    HA = np.array([
        [a[0], a[1], a[2]],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # Step 8: Compute H1
    H1 = HA @ H2 @ M

    return H1, H2



def compute_rectified_image(im: np.array, 
                            H: np.array) -> tuple:
    '''
    Rectifies an image using a homography matrix
    Arguments:
        im - an image
        H - a homography matrix that rectifies the image
    Returns:
        new_image - a new image matrix after applying the homography
        offset - the offest in the image.
    '''
    h, w = im.shape[:2]

    # נגדיר את 4 הפינות של התמונה המקורית
    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)

    # נעביר את הפינות דרך ההומוגרפיה
    corners_h = np.hstack([corners, np.ones((4, 1))])
    warped_corners = (H @ corners_h.T).T
    warped_corners = warped_corners[:, :2] / warped_corners[:, [2]]

    # נמצא את הגבולות החדשים של התמונה
    min_x = np.floor(np.min(warped_corners[:, 0])).astype(int)
    min_y = np.floor(np.min(warped_corners[:, 1])).astype(int)
    max_x = np.ceil(np.max(warped_corners[:, 0])).astype(int)
    max_y = np.ceil(np.max(warped_corners[:, 1])).astype(int)

    # גודל התמונה החדשה
    new_w = max_x - min_x
    new_h = max_y - min_y

    # נבצע שינוי (offset) כדי להזיז את כל התמונה החדשה כך שלא תצא מחוץ למסך
    offset_x = -min_x
    offset_y = -min_y

    # מטריצת offset הומוגנית
    offset_mat = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ])

    # נשלב את ההיסט עם ההומוגרפיה
    H_total = offset_mat @ H

    # נבצע את שינוי הפרספקטיבה בפועל
    new_image = cv2.warpPerspective(im, H_total, (new_w, new_h))

    return new_image, (offset_x, offset_y)


def find_matches(img1: np.array, img2: np.array) -> tuple:
    """
    Find matches between two images using SIFT
    Arguments:
        img1 - the first image
        img2 - the second image
    Returns:
        kp1 - the keypoints of the first image
        kp2 - the keypoints of the second image
        matches - the matches between the keypoints
    """
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 2. הגדרת התאמה בעזרת FLANN (מהיר)
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = flann.knnMatch(des1, des2, k=2)

    # 3. סינון התאמות גרועות (Lowe's ratio test)
    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

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
    # print("e1", e1)
    # print("e2", e2)
    assert np.allclose(e1, expected_e1, rtol=1e-2), f"e1 does not match this expected value:\n{expected_e1}"
    assert np.allclose(e2, expected_e2, rtol=1e-2), f"e2 does not match this expected value:\n{expected_e2}"
    np.save(env.p6.e1, e1)
    np.save(env.p6.e2, e2)

    # Part 6.b
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print("H1:\n", H1)
    print("H2:\n", H2)
    print("ex H1:\n", expected_H1)
    print("ex H2:\n", expected_H2)
    # assert np.allclose(H1, expected_H1, rtol=1e-2), f"H1 does not match this expected value:\n{expected_H1}"
    assert np.allclose(H2, expected_H2, rtol=1e-2), f"H2 does not match this expected value:\n{expected_H2}"
    # np.save(env.p6.H1, H1)
    # np.save(env.p6.H2, H2)
    # H1 =expected_H1
    # H2 =expected_H2
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
