#!/usr/bin/env python3
import ast
import numpy as np
import json

def parse_groundtruth(fname : str) -> dict:
    with open(fname,'r') as f:
        gt_dict = ast.literal_eval(f.readline())
        
        aruco_dict = {}
        for key in gt_dict:
            aruco_dict[key] = np.reshape([gt_dict[key]["x"], gt_dict[key]["y"]], (2,1))
    return aruco_dict

def parse_user_fruit(fname : str) -> dict:
    with open(fname, 'r') as f:
        usr_dict = ast.literal_eval(f.read())
        fruit_dict = {}
        for f in usr_dict:
            fruit_dict[f] = np.reshape([usr_dict[f][0],usr_dict[f][1]], (2,1))
    return fruit_dict

def match_aruco_points(aruco0 : dict, aruco1 : dict):
    missing_fruit = 0
    points0 = []
    points1 = []
    for key in aruco0:
        if not key in aruco1:
            missing_fruit+=1
            continue
        
        points0.append(aruco0[key])
        points1.append(aruco1[key])
    return np.hstack(points0), np.hstack(points1), missing_fruit

def parse_user_fruit(fname : str) -> dict:
    with open(fname, 'r') as f:
        usr_dict = ast.literal_eval(f.read())
        fruit_dict = {}
        for f in usr_dict:
            fruit_dict[f] = np.reshape([usr_dict[f][0],usr_dict[f][1]], (2,1))
    return fruit_dict

def solve_umeyama2d(points1, points2):
    # Solve the optimal transform such that
    # R(theta) * p1_i + t = p2_i

    assert(points1.shape[0] == 2)
    assert(points1.shape[0] == points2.shape[0])
    assert(points1.shape[1] == points2.shape[1])


    # Compute relevant variables
    num_points = points1.shape[1]
    mu1 = 1/num_points * np.reshape(np.sum(points1, axis=1),(2,-1))
    mu2 = 1/num_points * np.reshape(np.sum(points2, axis=1),(2,-1))
    sig1sq = 1/num_points * np.sum((points1 - mu1)**2.0)
    sig2sq = 1/num_points * np.sum((points2 - mu2)**2.0)
    Sig12 = 1/num_points * (points2-mu2) @ (points1-mu1).T @ np.linalg.pinv((points1-mu1)@(points1-mu1).T)

    # Use the SVD for the rotation
    U, d, Vh = np.linalg.svd(Sig12)
    S = np.eye(2)
    if np.linalg.det(Sig12) < 0:
        S[-1,-1] = -1
    
    # Return the result as an angle and a 2x1 vector
    R = U @ S @ Vh
    theta = np.arctan2(R[1,0],R[0,0])
    x = mu2 - R @ mu1

    return theta, x

def apply_transform(theta, x, points):
    # Apply an SE(2) transform to a set of 2D points
    assert(points.shape[0] == 2)
    
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    points_transformed =  R @ points + x
    return points_transformed


def compute_rmse(points1, points2):
    # Compute the RMSE between two matched sets of 2D points.
    assert(points1.shape[0] == 2)
    assert(points1.shape[0] == points2.shape[0])
    assert(points1.shape[1] == points2.shape[1])
    num_points = points1.shape[1]
    residual = (points1-points2).ravel()
    MSE = 1.0/num_points * np.sum(residual**2)

    return np.sqrt(MSE)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Solve the RVSS alignment problem")
    parser.add_argument("groundtruth", type=str, help="The ground truth file name.")
    parser.add_argument("estimate", type=str, help="The estimate file name.")
    args = parser.parse_args()

    gt_aruco = parse_groundtruth(args.groundtruth)
    us_aruco = parse_user_fruit(args.estimate)

    us_vec, gt_vec, missing = match_aruco_points(us_aruco, gt_aruco)


    rmse = compute_rmse(us_vec, gt_vec)
    print("The RMSE before alignment: {}".format(rmse))

    theta, x = solve_umeyama2d(us_vec, gt_vec)
    us_vec_aligned = apply_transform(theta, x, us_vec)

    print("The following parameters optimally transform the estimated points to the ground truth.")
    print("Rotation Angle: {}".format(theta))
    print("Translation Vector: ({}, {})".format(x[0,0], x[1,0]))

    rmse = compute_rmse(us_vec_aligned, gt_vec)
    print("Failed to detect {} kinds of fruits".format(missing))
    print("The RMSE after alignment: {}".format(rmse))


