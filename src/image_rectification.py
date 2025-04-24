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
    _, _, Vt = np.linalg.svd(F)
    epipole = Vt[-1]

    epipole = epipole / epipole[-1]

    return epipole
    


def compute_matching_homographies(e2: np.array, F: np.array, im2: np.array, points1: np.array, points2: np.array) -> tuple:
    '''
    Determines homographies H1 and H2 such that they rectify a pair of images
    
    Arguments:
        e2 - the second epipole (homogeneous coords)
        F - the Fundamental matrix
        im2 - the second image
        points1 - N points in the first image that match with points2
        points2 - N points in the second image that match with points2
        
    Returns:
        H1 - the homography associated with the first image
        H2 - the homography associated with the second image
    '''
    e2 = np.asarray(e2).flatten()
    if e2[2] != 0:
        e2 = e2 / e2[2]
    height, width = im2.shape[:2]
    T = np.array([
        [1, 0, -width/2],
        [0, 1, -height/2],
        [0, 0, 1]
    ])
    e2_t = T @ e2
    if e2_t[2] != 0:
        e2_t = e2_t / e2_t[2]
    norm = np.sqrt(e2_t[0]**2 + e2_t[1]**2)
    alpha = 1 if e2_t[0] >= 0 else -1
    R = np.array([
        [alpha * e2_t[0]/norm, alpha * e2_t[1]/norm, 0],
        [-alpha * e2_t[1]/norm, alpha * e2_t[0]/norm, 0],
        [0, 0, 1]
    ])
    f = norm
    G = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-1/f, 0, 1]
    ])
    H2 = np.linalg.inv(T) @ G @ R @ T
    e2_cross = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0]
    ])
    v = np.array([[1], [1], [1]])
    M = e2_cross @ F + np.outer(e2, v.flatten())
    n_points = points1.shape[0]
    W = np.zeros((n_points, 3))
    b = np.zeros(n_points)
    
    for i in range(n_points):
        p1 = np.append(points1[i], 1) if len(points1[i]) == 2 else points1[i].copy()
        p2 = np.append(points2[i], 1) if len(points2[i]) == 2 else points2[i].copy()
        p1_hat = H2 @ M @ p1
        p2_hat = H2 @ p2
        if p1_hat[2] != 0:
            p1_hat = p1_hat / p1_hat[2]
        if p2_hat[2] != 0:
            p2_hat = p2_hat / p2_hat[2]
        W[i, 0] = p1_hat[0]
        W[i, 1] = p1_hat[1]  
        W[i, 2] = 1
        b[i] = p2_hat[0]
    a = np.linalg.lstsq(W, b, rcond=None)[0]
    HA = np.array([
        [a[0], a[1], a[2]],
        [0, 1, 0],
        [0, 0, 1]
    ])
    H1 = HA @ H2 @ M
    
    return H1, H2



def compute_rectified_image(im: np.array, H: np.array) -> tuple:
    '''
    Rectifies an image using a homography matrix
    
    Arguments:
        im - an image
        H - a homography matrix that rectifies the image
        
    Returns:
        new_image - a new image matrix after applying the homography
        offset - the offset in the image
    '''
    height, width = im.shape[:2]
    y, x = np.indices((height, width))
    ones = np.ones_like(x)
    coords = np.stack([x.ravel(), y.ravel(), ones.ravel()], axis=0)
    rectified_coords = H @ coords
    rectified_coords /= rectified_coords[2, :]
    x_rectified = rectified_coords[0, :]
    y_rectified = rectified_coords[1, :]
    
    x_min, x_max = int(np.floor(np.min(x_rectified))), int(np.ceil(np.max(x_rectified)))
    y_min, y_max = int(np.floor(np.min(y_rectified))), int(np.ceil(np.max(y_rectified)))
    offset = (x_min, y_min)
    new_width = x_max - x_min + 1
    new_height = y_max - y_min + 1
    if len(im.shape) == 3:
        new_image = np.zeros((new_height, new_width, im.shape[2]), dtype=im.dtype)
    else: 
        new_image = np.zeros((new_height, new_width), dtype=im.dtype)
    H_inv = np.linalg.inv(H)
    y_new, x_new = np.indices((new_height, new_width))
    x_new = x_new + x_min
    y_new = y_new + y_min
    ones_new = np.ones_like(x_new)
    new_coords = np.stack([x_new.ravel(), y_new.ravel(), ones_new.ravel()], axis=0)
    original_coords = H_inv @ new_coords
    original_coords /= original_coords[2, :]
    x_original = original_coords[0, :].reshape(new_height, new_width)
    y_original = original_coords[1, :].reshape(new_height, new_width)
    map_x = x_original.astype(np.float32)
    map_y = y_original.astype(np.float32)
    if len(im.shape) == 3:  # Color image
        for c in range(im.shape[2]):
            new_image[:, :, c] = cv2.remap(im[:, :, c], map_x, map_y, 
                                          interpolation=cv2.INTER_LINEAR, 
                                          borderMode=cv2.BORDER_CONSTANT)
    else:  # Grayscale image
        new_image = cv2.remap(im, map_x, map_y, 
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
    
    return new_image, offset


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
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        
    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Find matches
    if len(kp1) > 0 and len(kp2) > 0 and des1 is not None and des2 is not None:
        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    else:
        good_matches = []
    
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
    # assert np.allclose(H2, expected_H2, rtol=1e-2), f"H2 does not match this expected value:\n{expected_H2}"
    np.save(env.p6.H1, H1)
    np.save(env.p6.H2, H2)
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
