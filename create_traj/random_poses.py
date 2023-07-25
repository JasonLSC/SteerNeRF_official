import numpy as np
import glob
import os

def to_sphere(u, v):
    theta = 2 * np.pi * u
    phi = np.arccos(1 - 2 * v)
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)
    s = np.stack([cx, cy, cz])
    return s

def sample_on_sphere(range_u=(0, 1), range_v=(0, 1)):
    u = np.random.uniform(*range_u)
    v = np.random.uniform(*range_v)
    return to_sphere(u, v)

def look_at(eye, at=np.array([0, 0, 0]), up=np.array([0, 0, -1]), eps=1e-5):
    at = at.astype(float).reshape(1, 3)
    up = up.astype(float).reshape(1, 3)

    eye = eye.reshape(-1, 3)
    up = up.repeat(eye.shape[0] // up.shape[0], axis=0)
    eps = np.array([eps]).reshape(1, 1).repeat(up.shape[0], axis=0)

    #z_axis = eye - at
    z_axis = at - eye
    z_axis /= np.max(np.stack([np.linalg.norm(z_axis, axis=1, keepdims=True), eps]))

    x_axis = np.cross(up, z_axis)
    x_axis /= np.max(np.stack([np.linalg.norm(x_axis, axis=1, keepdims=True), eps]))

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.max(np.stack([np.linalg.norm(y_axis, axis=1, keepdims=True), eps]))

    r_mat = np.concatenate((x_axis.reshape(-1, 3, 1), y_axis.reshape(-1, 3, 1), z_axis.reshape(-1, 3, 1)), axis=2)

    return r_mat

def sample_pose(radius, range_u=(0, 1), range_v=(0, 1)):
    # sample location on unit sphere
    loc = sample_on_sphere(range_u, range_v)
    
    if isinstance(radius, tuple):
        radius = np.random.uniform(*radius)

    loc = loc * radius
    R = look_at(loc)[0]

    RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
    RT = np.concatenate([RT, np.array([0.,0.,0.,1.]).reshape(1,4)], axis=0)
    return RT

def sample_sphere_pose(radius, N_sample, range_u=(0, 1), range_v=(0, 1)):
    RT_list = []
    u_list = np.linspace(range_u[0], range_u[1], num=N_sample)
    v_list = np.linspace(range_v[0], range_v[1], num=N_sample)
    for u, v in zip(u_list, v_list):
        # sequentially sample location on unit sphere
        loc = to_sphere(u, v)
    
        if isinstance(radius, tuple):
            radius = np.random.uniform(*radius)

        loc = loc * radius
        R = look_at(loc)[0]

        RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
        RT = np.concatenate([RT, np.array([0.,0.,0.,1.]).reshape(1,4)], axis=0)
        RT_list.append(RT)

    return RT_list

def get_radius(poses):
    T = poses[:, :3, 3]
    radius = np.linalg.norm(T, axis=1)
    return np.mean(radius)

def load_poses(root, dataset='', objectname=''):
    if dataset=='Synthetic_NSVF' or dataset=='Synthetic_NeRF':
        pose_files = glob.glob(os.path.join(root, dataset, objectname, 'pose', '2_*.txt'))
        poses = [np.loadtxt(f) for f in pose_files] 
        poses = np.concatenate(poses).reshape(-1,4,4)
    elif dataset=='BlendedMVS' or dataset=='':
        poses = np.loadtxt(os.path.join(root, dataset, objectname, 'test_traj.txt'))
        poses = np.reshape(poses, (-1, 4, 4))
    elif dataset=='TanksAndTemple':
        pose_files = glob.glob(os.path.join(root, dataset, objectname, 'pose', '0_*.txt'))
        pose_files = sorted(pose_files)
        poses = [np.loadtxt(f) for f in pose_files] 
        poses = np.concatenate(poses).reshape(-1,4,4)
    else:
        raise RuntimeError('Unknown dataset!')
    return poses


if __name__=="__main__":
    root = '/is/rg/avg/creiser/anti/data/nsvf'
    dataset = 'Synthetic_NSVF'
    objectname = 'Bike'
    poses = load_poses(root, dataset, objectname)
    if dataset=='Synthetic_NeRF' or dataset=='Synthetic_NSVF':
        np.savetxt(os.path.join('output', dataset, objectname, 'test_traj.txt'), poses.reshape(-1,4))
    radius = get_radius(poses)
    print('Mean radius from test trajectories: %f' % radius)

    N_sample = 200
    out_file = os.path.join('output', dataset, objectname, 'random_traj.txt')
    RT = [sample_pose(radius, range_v=(0.0,0.5)) for _ in range(N_sample)]
    RT = np.concatenate(RT)
    np.savetxt(out_file, RT)

