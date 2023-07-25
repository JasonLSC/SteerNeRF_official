import pdb

import open3d as o3d
import os
import matplotlib.pyplot as plt
import numpy as np
from random_poses import load_poses, sample_pose, sample_sphere_pose
import copy
from bezier_curve import bezier_curve
from scipy import interpolate
from datetime import datetime


if __name__=='__main__':

    N_sample = 60

    root = '/work/Users/lisicheng/Dataset/'
    dataset = 'TanksAndTemple'
    objectname = 'Truck'
    pose_id_list = [33, 34, 32, 21, 14]
    # pose_id_list = [4, 32, 38, 53, 42]

    save_dir = '/work/Users/lisicheng/Dataset/nerf_sr_data/TanksAndTemple/Truck/'

    poses = load_poses(root, dataset, objectname)
    
    RT_all = []

    for i in range(len(pose_id_list) - 1):
        # if i == 0:
        #     ind = [pose_id_list[i+1], pose_id_list[i]]
        # else:
        ind = [pose_id_list[i], pose_id_list[i+1]]
        RT = [poses[i] for i in ind]
        # pdb.set_trace()
        # print(RT)
        interpolated_pose = bezier_curve(RT, num_steps=N_sample)
        RT = np.concatenate(interpolated_pose)
        RT_all.append(RT)
    
    RT_all = np.concatenate(RT_all)
    poses = RT_all.reshape(-1, 4, 4)

    # Returns a datetime object containing the local date and time
    dateTimeObj = datetime.now()
    dateObj = dateTimeObj.date()
    timeObj = dateTimeObj.time()
    timeStr = f'{dateObj.year}_{dateObj.month}_{dateObj.day}_{timeObj.hour}_{timeObj.minute}_{timeObj.second}'
    print(timeStr)
    # get the time object from datetime object
    np.savetxt(save_dir+f'test_traj_{timeStr}.txt', poses.reshape(-1, 4))

    # textured_mesh = o3d.io.read_triangle_mesh("test_data/crate/crate.obj")
    # custom_draw_geometry_with_camera_trajectory(textured_mesh, poses)
