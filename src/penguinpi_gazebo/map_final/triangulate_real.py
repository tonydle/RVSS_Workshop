import numpy as np
# import cv2
import math
from machinevisiontoolbox import Image

import matplotlib.pyplot as plt

def triangulate_position(image_plane_locations, camera_poses, camera_matrix):
    # image_plane_locations: (2, n)
    # camera_poses: (3, n)
    # camera_matrix: (3,3)
    n = image_plane_locations.shape[-1]
    assert image_plane_locations.shape == (2,n), image_plane_locations.shape
    assert camera_poses.shape == (3, n), camera_poses.shape
    assert camera_matrix.shape == (3, 3), camera_matrix.shape
    f = camera_matrix[0,0]
    Cx = camera_matrix[0,2]
    u_dash = ((image_plane_locations[0]-Cx)/f).reshape(1, -1) # (1, n)
    y = np.concatenate([np.ones(u_dash.shape), -u_dash], axis=0) # (2, n)
    y_hat = y / np.linalg.norm(y, axis=0) # (2, n)
    thetas = camera_poses[2] # (n,)
    R = [np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]]) for th in thetas]
    z_perp = np.array([[0, 1], [-1, 0]])
    A = np.zeros((n,2))
    b = np.zeros((n,))
    for i in range(n):
        A[i] = (y_hat[:,i].T @ R[i].T @ z_perp).reshape(1, 2)
        b[i] = A[i] @ camera_poses[:2,i]
    ret = np.linalg.lstsq(A, b) # (2,)
    P= ret[0]
    return P

def images_to_single_fruit_centroid(fruit_number, image_path):
    import PIL
    image = PIL.Image.open(image_path).resize((320,240), PIL.Image.NEAREST)
    # fruit = (Image(image_path, grey=True)==fruit_number)*1.0
    fruit = Image(image)==fruit_number
    # import pdb; pdb.set_trace()
    blobs = fruit.blobs()
    # plt.imshow(fruit.image)
    # plt.annotate(str(fruit_number), np.array(blobs[0].centroid).reshape(2,))
    # plt.show()
    # assert len(blobs) == 1, "Must have only 1 fruit of the given fruit type in this photo"
    return np.array(blobs[0].centroid).reshape(2,) # (2,)

def get_triangulation_params(base_dir):
    # Assume everything is in base_dir/workshop_output
    import os
    from pathlib import Path
    import ast
    fruit_lst_centroids = [[], [], [], []]
    fruit_lst_pose = [[], [], [], []]
    base_dir = Path(base_dir)

    files = os.listdir(base_dir/'workshop_output' )
    assert 'images.txt' in files

    image_poses = {}
    with open(base_dir/'workshop_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']
    
    for file_path in image_poses.keys():
        img_vals = set(Image(base_dir / file_path, grey=True).image.reshape(-1))
        for fruit_num in img_vals:
            if fruit_num>0:
                try:
                    centroid = images_to_single_fruit_centroid(fruit_num, base_dir/file_path)
                    pose = image_poses[file_path]
                    fruit_lst_centroids[fruit_num-1].append(centroid)
                    fruit_lst_pose[fruit_num-1].append(np.array(pose).reshape(3,))
                except ZeroDivisionError:
                    pass
    completed_triangulations = {}
    for i in range(4):
        if len(fruit_lst_centroids[i])>0:
            centroids = np.stack(fruit_lst_centroids[i], axis=1) # (2,n)
            pose = np.stack(fruit_lst_pose[i], axis=1) # (3,n)
            completed_triangulations[i+1] = {'centroids': centroids, 'pose': pose} # entroids (2,n), pose (3, n)
    return completed_triangulations

def run_triangulation(base_dir, camera_matrix):
    completed_triangulations = get_triangulation_params(base_dir)
    triangulations_dict = {}
    fruit_list = ['apple', 'banana', 'pear', 'lemon']
    for fruit_num in completed_triangulations.keys():
        centroids = completed_triangulations[fruit_num]['centroids']
        pose = completed_triangulations[fruit_num]['pose']
        triangulations_dict[fruit_list[fruit_num-1]] = triangulate_position(centroids, pose, camera_matrix).tolist()
    import json
    with open('fruit.txt', 'w') as fp:
        json.dump(triangulations_dict, fp)
    


if __name__ == "__main__":
    # image_plane_locations = np.ones((2,10))*5
    # camera_poses = np.ones((3,10)) * 3
    # camera_matrix = np.ones((3,3))/2
    
    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    run_triangulation('./', camera_matrix)
    print('Result saved in fruit.txt!')
        

    # print(P)


