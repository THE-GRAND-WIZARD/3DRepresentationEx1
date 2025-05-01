import os
import sys
import env
import src.utils.engine
import src.utils.utils as utils
import numpy as np
import cv2
from scipy.optimize import least_squares

from src.calibrate_camera import *
from src.image_rectification import *
from src.the_3D_reconstruction import *


def reprojection_error(params, n_cameras, n_points, camera_indices, point_indices, observed_2d, camera_matrix):
    """
    Compute reprojection error for all observations given camera and point parameters.
    """
    # Unpack
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d      = params[n_cameras * 6:].reshape((n_points, 3))

    residuals = []
    for obs_idx, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):
        # rotation (Rodrigues) and translation
        rvec = camera_params[cam_idx, :3]
        tvec = camera_params[cam_idx, 3:6]
        R, _ = cv2.Rodrigues(rvec)
        # projection matrix
        P = camera_matrix @ np.hstack((R, tvec.reshape(3,1)))
        # homogeneous point
        X = np.hstack((points_3d[pt_idx], 1.0))
        x_proj = P.dot(X)
        x_proj = x_proj[:2] / x_proj[2]
        residuals.extend(x_proj - observed_2d[obs_idx])
    return np.array(residuals)


def bundle_adjustment(camera_params, points_3d, camera_indices, point_indices, observed_2d, camera_matrix):
    """
    Run bundle adjustment to refine camera poses and 3D points.
    """
    n_cameras = camera_params.shape[0]
    n_points  = points_3d.shape[0]
    # pack initial parameters
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    # least squares
    result = least_squares(
        fun=reprojection_error,
        x0=x0,
        args=(n_cameras, n_points, camera_indices, point_indices, observed_2d, camera_matrix),
        method='lm'
    )
    opt = result.x
    cam_opt = opt[:n_cameras * 6].reshape((n_cameras, 6))
    pts_opt = opt[n_cameras * 6:].reshape((n_points, 3))
    return cam_opt, pts_opt


def main():
    # prepare output
    if not os.path.exists(env.p8.output):
        os.makedirs(env.p8.output)
    # fixed intrinsics
    _, _ = calibrate_camera_from_chessboard(env.p7.chessboard, (16,10))
    camera_matrix = np.eye(3)
    f = 719.5459
    camera_matrix[0,0] = f; camera_matrix[1,1] = f

    # load image list
    images = sorted([f for f in os.listdir(env.p8.statue_images)
                     if f.lower().endswith(('.png','.jpg','.jpeg'))])

    # store poses and 3D points
    camera_params = []  # each: [rvec(3), tvec(3)]
    cam0 = np.zeros(6)  # first camera at origin
    camera_params.append(cam0)

    camera_indices = []
    point_indices  = []
    observed_2d    = []
    all_points3d   = []
    total_pts = 0

    R_prev = np.eye(3)
    t_prev = np.zeros(3)

    # pairwise triangulation and correspondence collection
    for i in range(len(images)-1):
        im1 = utils.load_image(os.path.join(env.p8.statue_images, images[i]))
        im2 = utils.load_image(os.path.join(env.p8.statue_images, images[i+1]))
        kp1, kp2, matches = find_matches(im1, im2)
        F, mask, pts1, pts2 = recover_fundamental_matrix(kp1, kp2, matches)
        in1, in2 = get_inliers(mask, pts1, pts2)
        E = compute_essential_matrix(camera_matrix, F)
        Rs, Ts = estimate_initial_RT(E)
        R_loc, t_loc = find_best_RT(Rs, Ts, in1, in2, camera_matrix)

        # triangulate local
        P1 = get_identity_projection_matrix(camera_matrix)
        P2 = get_local_projection_matrix(camera_matrix, R_loc, t_loc)
        pts4d = cv2.triangulatePoints(P1, P2, in1, in2)
        pts3d_local = (pts4d[:3] / pts4d[3]).T

        # global transform
        pts3d_global = (R_prev @ pts3d_local.T).T + t_prev
        all_points3d.append(pts3d_global)

        # register correspondences
        n = pts3d_global.shape[0]
        for j in range(n):
            # observation in camera i
            camera_indices.append(i)
            point_indices.append(total_pts + j)
            observed_2d.append(in1[:, j])
            # observation in camera i+1
            camera_indices.append(i+1)
            point_indices.append(total_pts + j)
            observed_2d.append(in2[:, j])
        total_pts += n

        # update global pose
        R_prev = R_prev @ R_loc
        t_prev = R_prev @ t_loc.flatten() + t_prev
        rvec, _ = cv2.Rodrigues(R_prev)
        camera_params.append(np.hstack((rvec.flatten(), t_prev)))

        print(f"Triangulated pair {i}->{i+1}: {n} points.")

    if not all_points3d:
        print("No points to bundle-adjust.")
        return

    # stack data
    points3d = np.vstack(all_points3d)
    camera_params = np.array(camera_params)
    camera_indices = np.array(camera_indices, dtype=int)
    point_indices  = np.array(point_indices,  dtype=int)
    observed_2d    = np.array(observed_2d,    dtype=float)

    print(f"Running BA on {camera_params.shape[0]} cameras and {points3d.shape[0]} points ({observed_2d.shape[0]} observations)...")
    cam_opt, pts_opt = bundle_adjustment(
        camera_params, points3d,
        camera_indices, point_indices,
        observed_2d, camera_matrix
    )
    show_points_matplotlib(pts_opt)


if __name__ == '__main__':
    main()
