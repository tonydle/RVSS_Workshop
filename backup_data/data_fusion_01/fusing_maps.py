#!/usr/bin/env python3
import ast
import numpy as np
import json

def parse_groundtruth(fname : str) -> dict:
    with open(fname,'r') as f:
        gt_dict = ast.literal_eval(f.readline())
        
        aruco_dict = {}
        for key in gt_dict:
            if key.startswith("aruco"):
                aruco_num = int(key[5])
                aruco_dict[aruco_num] = np.reshape([gt_dict[key]["x"], gt_dict[key]["y"]], (2,1))
    return aruco_dict

# def parse_user_map(fname : str) -> dict:
#     with open(fname, 'r') as f:
#         usr_dict = ast.literal_eval(f.read())
#         aruco_dict = {}
#         for (i, tag) in enumerate(usr_dict["taglist"]):
#             aruco_dict[tag] = np.reshape([usr_dict["map"][0][i],usr_dict["map"][1][i]], (2,1))
#     return aruco_dict

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def parse_user_map(fname : str) -> dict:
    with open(fname, 'r') as f:
        usr_dict = ast.literal_eval(f.read())
        aruco_dict = {}
        for (i, tag) in enumerate(usr_dict["taglist"]):
            aruco_dict[tag] = np.reshape([usr_dict["map"][0][i],usr_dict["map"][1][i]], (2,1))
    return aruco_dict

def parse_user_fruit(fname : str) -> dict:
    with open(fname, 'r') as f:
        usr_dict = ast.literal_eval(f.read())
        fruit_dict = {}
        for f in usr_dict:
            fruit_dict[f] = np.reshape([usr_dict[f][0],usr_dict[f][1]], (2,1))
    return fruit_dict

def dict2nparr(fruit_dict):
    points = []
    for key in fruit_dict:   
        points.append(fruit_dict[key])
    return np.hstack(points)

def match_aruco_points(aruco0 : dict, aruco1 : dict):
    points0 = []
    points1 = []
    for key in aruco0:
        if not key in aruco1:
            continue
        
        points0.append(aruco0[key])
        points1.append(aruco1[key])
    return np.hstack(points0), np.hstack(points1)

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
    # sig_inv = np.linalg.inv(points1-mu1.T @ points1-mu1)
    # Sig12 = 1/num_points * (points2-mu2) @ (points1-mu1).T @ sig_inv
    Sig12 = 1/num_points * (points2-mu2) @ (points1-mu1).T
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

def apply_transform(theta, x, fruit_dict):
    # Apply an SE(2) transform to a set of 2D points
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    for key in fruit_dict:
        point = fruit_dict[key]
        point = point.reshape(2, 1)
        assert(point.shape[0] == 2)
        fruit_dict[key] =  R @ point + x
    return fruit_dict

def compute_rmse(points1, points2):
    # Compute the RMSE between two matched sets of 2D points.
    assert(points1.shape[0] == 2)
    assert(points1.shape[0] == points2.shape[0])
    assert(points1.shape[1] == points2.shape[1])
    num_points = points1.shape[1]
    residual = (points1-points2).ravel()
    MSE = 1.0/num_points * np.sum(residual**2)

    return np.sqrt(MSE)

def fuse_fruit_map(fruit_map_1, fruit_map_2):
    label = ['apple', 'banana', 'pear', 'lemon']
    fused_map = {}
    for fruit in label:
        point = np.zeros((2, 1))
        occurance = 0
        if fruit in fruit_map_1.keys():
            point += fruit_map_1[fruit]
            occurance += 1 
        if fruit in fruit_map_2.keys():
            point += fruit_map_2[fruit]
            occurance += 1 
        point /= occurance
        fused_map[fruit] = point.tolist()
    return fused_map

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Solve the RVSS alignment problem")
    parser.add_argument("map_1", type=str, help="The estimate map 1 file name.")
    parser.add_argument("fruit_1", type=str, help="The estimate fruit 1 file name.")
    parser.add_argument("map_2", type=str, help="The estimate map 2 file name.")
    parser.add_argument("fruit_2", type=str, help="The estimate fruit 2 file name.")
    args = parser.parse_args()

    slam1 = parse_user_map(args.map_1)
    slam2 = parse_user_map(args.map_2)
    fruit_map_1 = parse_user_fruit(args.fruit_1)
    fruit_map_2 = parse_user_fruit(args.fruit_2)

    map1 = Merge(slam1, fruit_map_1)
    map2 = Merge(slam2, fruit_map_2)

    map1_vec, map2_vec = match_aruco_points(map1, map2)

    rmse = compute_rmse(map1_vec, map2_vec)
    # print("The RMSE before alignment: {}".format(rmse))

    theta, x = solve_umeyama2d(map1_vec, map2_vec)
    fruit_map_1_aligned = apply_transform(theta, x, fruit_map_1)
    
    fused_fruit_map = fuse_fruit_map(fruit_map_1_aligned, fruit_map_2)

    print(fruit_map_1_aligned, '\n', fruit_map_2, '\n', fused_fruit_map)

    import json
    with open('result.txt', 'w') as fp:
        json.dump(fused_fruit_map, fp)

    print("The following parameters optimally transform the estimated points to the ground truth.")
    print("Rotation Angle: {}".format(theta))
    print("Translation Vector: ({}, {})".format(x[0,0], x[1,0]))



