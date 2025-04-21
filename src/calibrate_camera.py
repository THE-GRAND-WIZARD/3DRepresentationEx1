import os
import sys
import env
import src.utils.utils as utils

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import math


def get_3D_object_points(chessboard_size: tuple) -> np.ndarray:
    """
    Get the 3D object points of a chessboard
    Args:
        chessboard_size: Tuple containing the number of columns and rows in the chessboard
    Returns:
        Numpy array containing the 3D object points
    """
    objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    return objp


def undistort_image(image: np.ndarray,
                    camera_matrix: np.ndarray,
                    dist_coeffs: np.ndarray) -> np.ndarray:
    """
    Undistort an image
    Args:
        image: Numpy array containing the image
        camera_matrix: Numpy array containing the camera matrix
        dist_coeffs: Numpy array containing the distortion coefficients
    Returns:
        Numpy array containing the undistorted image
    """
    h, w = image.shape[:2]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    k1, k2, p1, p2, k3 = dist_coeffs.ravel()

    # Generate coordinate grid
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_norm = (x - cx) / fx
    y_norm = (y - cy) / fy
    r2 = x_norm ** 2 + y_norm ** 2
    r4 = r2 ** 2
    r6 = r2 * r4

    # Apply distortion model
    x_dist = x_norm * (1 + k1 * r2 + k2 * r4 + k3 * r6) + 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm ** 2)
    y_dist = y_norm * (1 + k1 * r2 + k2 * r4 + k3 * r6) + 2 * p2 * x_norm * y_norm + p1 * (r2 + 2 * y_norm ** 2)

    # Convert back to pixel coordinates
    x_pix = (x_dist * fx + cx).flatten()
    y_pix = (y_dist * fy + cy).flatten()

    # Interpolate using scipy
    if image.ndim == 3:
        channels = []
        for i in range(3):
            interp = map_coordinates(image[:, :, i], [y_pix, x_pix], order=1, mode='reflect').reshape(h, w)
            channels.append(interp)
        undistorted = np.stack(channels, axis=-1).astype(np.uint8)
    else:
        undistorted = map_coordinates(image, [y_pix, x_pix], order=1, mode='reflect').reshape(h, w).astype(np.uint8)

    return undistorted


def load_grayscale_image(image: np.ndarray) -> np.ndarray:
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    return gray_image


def calibrate_camera(object_points: np.ndarray,
                     corners: np.ndarray,
                     image_size: tuple) -> tuple:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        [object_points], [corners], image_size, None, None
    )

    return camera_matrix, dist_coeffs


def find_chessboard_corners(image: np.ndarray, chessboard_size: tuple) -> np.ndarray:
    ret, corners = cv2.findChessboardCorners(image, chessboard_size, None)

    if ret is False:
        raise ValueError("Verify correct dimensions of chessboard")

    return corners


def refine_corners(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)

    return corners


def draw_corners(image: np.ndarray, chessboard_size: tuple, corners: np.ndarray):
    cv2.drawChessboardCorners(image, chessboard_size, corners, True)
    plt.imshow(image)
    plt.title("Chessboard Corners")
    plt.show()


if __name__ == "__main__":
    if not os.path.exists(env.p4.output):
        os.makedirs(env.p4.output)
    expected_camera_matrix = np.load(env.p4.expected_camera_matrix)
    expected_dist_coeffs = np.load(env.p4.expected_dist_coeffs)

    # Part 4.a - Compute ideal intrinsic matrix
    image_width, image_height = 1280, 720  # Assumed size
    FoV = math.radians(45)
    f_r = 1 / (2 * math.tan(FoV / 2))
    focal_length = f_r * min(image_height, image_width)
    c_x, c_y = image_width / 2, image_height / 2
    ideal_intrinsic_matrix = np.array([
        [focal_length, 0, c_x],
        [0, focal_length, c_y],
        [0, 0, 1]
    ])

    # Part 4.b
    chessboard_size = (14, 9)  # (columns, rows)

    image = utils.load_image(env.p3.chessboard_path)
    grayscale_image = load_grayscale_image(image)
    corners = find_chessboard_corners(grayscale_image, chessboard_size)
    corners = refine_corners(grayscale_image, corners)
    draw_corners(image, chessboard_size, corners)
    Image.fromarray(image).save(env.p4.chessboard_corners)

    # Part 4.c
    object_points = get_3D_object_points(chessboard_size)
    camera_matrix, dist_coeffs = calibrate_camera(object_points, corners, grayscale_image.shape[::-1])
    print("Camera Matrix:")
    print(camera_matrix)
    assert np.allclose(camera_matrix, expected_camera_matrix,
                       atol=1e-2), f"Camera matrix does not match this expected matrix:\n{expected_camera_matrix}"
    np.save(env.p4.camera_matrix, camera_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeffs)
    assert np.allclose(dist_coeffs, expected_dist_coeffs,
                       atol=1e-2), f"Distortion coefficients do not match these expected coefficients:\n{expected_dist_coeffs}"
    np.save(env.p4.dist_coeff, dist_coeffs)

    # Part 4.d
    undistorted_image = undistort_image(image, camera_matrix, dist_coeffs)
    plt.imshow(undistorted_image)
    plt.title("Undistorted Image")
    plt.show()
    Image.fromarray(undistorted_image).save(env.p4.undistorted_image)