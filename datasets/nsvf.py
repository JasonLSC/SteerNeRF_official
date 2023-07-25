import pdb

import torch
import glob
import numpy as np
import os
from tqdm import tqdm
from einops import rearrange

from .ray_utils import get_ray_directions
from .color_utils import read_image, read_mask

from .base import BaseDataset


class NSVFDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()
        self.frame_num = kwargs.get('frame_num')
        self.training_stage = kwargs.get('training_stage')

        if split=='train': #or 'Tanks' in root_dir
            self.preload = False
        else:
            self.preload = True

        if 'Tanks' in root_dir:
            self.preload = True

        if kwargs.get('val_only'):
            self.preload = False

        if kwargs.get('training_stage', 'unknown') == 'NeRF_pretrain':
            self.preload = True

        if kwargs.get('read_meta', True):
            xyz_min, xyz_max = \
                np.loadtxt(os.path.join(root_dir, 'bbox.txt'))[:6].reshape(2, 3)
            self.shift = (xyz_max+xyz_min)/2
            self.scale = (xyz_max-xyz_min).max()/2 * 1.05 # enlarge a little

            # hard-code fix the bound error for some scenes...
            if 'Mic' in self.root_dir: self.scale *= 1.2
            elif 'Lego' in self.root_dir: self.scale *= 1.1

            self.read_meta(split)

        # TODO: write into BaseDataset
        # print(f"Get {kwargs.get('super_sampling_factor')}")
        if kwargs.get('super_sampling_factor'):
            if split=='test':
                print("test into low res")
            self.super_sampling_factor = float(kwargs['super_sampling_factor'])
            # print(f"self.super_sample_factor {self.super_sampling_factor}")
            low_res_K = self.K

            # print(f"intrinsic:{self.K}")
            low_res_K[:2] *= (1/self.super_sampling_factor)
            low_res_K = low_res_K.cpu().numpy()
            # get low res from intrinsic
            # But it falls when "TanksAndTemple"
            # low_res_w, low_res_h = low_res_K[:2, -1] * 2

            low_res_w = int(self.img_wh[0] * (1 / self.super_sampling_factor))
            low_res_h = int(self.img_wh[1] * (1 / self.super_sampling_factor))
            # low_res_w = low_res_w.astype(int)
            # low_res_h = low_res_h.astype(int)
            self.low_res_K = low_res_K

            print(f"down sample res:{low_res_h, low_res_w} for super sample")

            # overwrite self.directions
            self.directions = get_ray_directions(low_res_h, low_res_w, low_res_K)
            self.low_res_h = low_res_h
            self.low_res_w = low_res_w
            # pdb.set_trace()
            self.high_res_h = int(low_res_h * self.super_sampling_factor)
            self.high_res_w = int(low_res_w * self.super_sampling_factor)


            # crop patch from low_res img
            #self.patch_res_h, self.patch_res_w = (150, 150)
            patch_res_h = patch_res_w = kwargs.get('patch_size')
            self.patch_res_h, self.patch_res_w = (patch_res_h, patch_res_w)
            self.patch_high_res_h = int(self.patch_res_h * self.super_sampling_factor) # used to SR output reshape
            self.patch_high_res_w = int(self.patch_res_w * self.super_sampling_factor)

            # self.rays_low for low res supervision
            # if self.preload:
            #     factor = int(self.super_sampling_factor)
            #     rays = np.reshape(self.rays, [self.rays.shape[0], self.img_wh[1], self.img_wh[0], -1])
            #     rays_low = block_reduce(rays, block_size=(1, factor, factor, 1), func=np.mean)
            #     self.rays_low = rearrange(rays_low, 'b h w c -> b (h w) c')
            #     print(f"shape of rays low:{rays_low.shape}")

        print(f"Shape of directions in NSVFDataset:{self.directions.shape}")
        # print(f"NSVFDataset has attribute:{hasattr(self, 'super_sampling_factor')}")

    def read_intrinsics(self):
        if 'Synthetic' in self.root_dir or 'Ignatius' in self.root_dir:
            with open(os.path.join(self.root_dir, 'intrinsics.txt')) as f:
                fx = fy = float(f.readline().split()[0]) * self.downsample
            if 'Synthetic' in self.root_dir:
                w = h = int(800*self.downsample)
            else:
                w, h = int(1920*self.downsample), int(1080*self.downsample)

            K = np.float32([[fx, 0, w/2],
                            [0, fy, h/2],
                            [0,  0,   1]])
        else:
            K = np.loadtxt(os.path.join(self.root_dir, 'intrinsics.txt'),
                           dtype=np.float32)[:3, :3]
            if 'BlendedMVS' in self.root_dir:
                w, h = int(768*self.downsample), int(576*self.downsample)
            elif 'Tanks' in self.root_dir:
                w, h = int(1920*self.downsample), int(1080*self.downsample)
            K[:2] *= self.downsample

        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        print(f"num of dirs:{self.directions.shape}")
        self.img_wh = (w, h)

    def read_meta(self, split):
        if self.preload:
            self.rays = []
            self.masks = []
        self.poses = []

        if split == 'test_traj': # BlendedMVS and TanksAndTemple
            if 'Ignatius' in self.root_dir:
                poses_path = \
                    sorted(glob.glob(os.path.join(self.root_dir, 'test_pose/*.txt')))
                poses = [np.loadtxt(p) for p in poses_path]
            else:
                poses = np.loadtxt(os.path.join(self.root_dir, 'test_traj.txt'))
                poses = poses.reshape(-1, 4, 4)
            for pose in poses:
                c2w = pose[:3]
                # c2w[:, 0] *= -1 # [left down front] to [right down front]
                c2w[:, 3] -= self.shift
                c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
                self.poses += [c2w]

            pose_seq_set = []
            for i in range(self.frame_num):
                pose_seq = []
                for j in range(self.frame_num):
                    idx = max(i-j, 0)
                    pose_seq.append(self.poses[idx])
                pose_seq_set.append(pose_seq)

            print(len(self.poses))
            for i in range(self.frame_num, len(self.poses)):
                pose_seq_set.append(self.poses[i: i-self.frame_num:-1])

            self.poses = pose_seq_set
        else:
            if split == 'train': prefix = '[0]' #'0_'
            elif split == 'trainval': prefix = '[0-1]_'
            elif split == 'trainvaltest': prefix = '[0-2]_'
            elif split == 'testvaltrain': prefix = '[0-2]_'
            elif split == 'val': prefix = '1_'
            elif split == 'testtrain': prefix = '0_' # debug via overfitting
            elif 'Synthetic' in self.root_dir: prefix = '2_' # test set for synthetic scenes
            elif split == 'test': prefix = '1_' # test set for real scenes
            elif split == 'test_no_gt': prefix = '0_'  # multiframe
            else: raise ValueError(f'{split} split not recognized!')
            print(f"Prefix:{prefix}")
            if split != 'test_no_gt':
                if 'Synthetic' in self.root_dir and self.training_stage == 'NeRF_pretrain': # Synthetic_NeRF pretrain
                    img_paths = sorted(glob.glob(os.path.join(self.root_dir, 'rgb', prefix + '*-00.png')))
                else:
                    img_paths = sorted(glob.glob(os.path.join(self.root_dir, 'rgb', prefix+'*.png')))
                img_paths = [pt for pt in img_paths if '_mask.png' not in pt]
                self.img_paths = img_paths

                if 'Synthetic' in self.root_dir and self.training_stage == 'NeRF_pretrain':  # Synthetic_NeRF pretrain
                    poses = sorted(glob.glob(os.path.join(self.root_dir, 'pose', prefix + '*-00-*.txt')))
                else:
                    poses = sorted(glob.glob(os.path.join(self.root_dir, 'pose', prefix+'*.txt')))
                assert(float(len(poses))/float(len(self.img_paths))==int(len(poses)/len(self.img_paths)))
                n_clip = int(len(poses)/len(self.img_paths))
                self.n_clip = n_clip
                print(f"n_clip: {n_clip}")
                print(f"frame_num: {self.frame_num}")

                poses_set = []
                for i in range(len(img_paths)):
                    poses_set.append(poses[i*n_clip:i*n_clip+self.frame_num])

                # Only Synthetic_NeRF use previous frames, TanksAndTemple use self-defined preceding frames
                if 'Synthetic' in self.root_dir:
                    # use previous frames
                    if split=='test':
                        for i in range(len(img_paths)-(self.frame_num-1), len(img_paths)):
                            poses_set[i] = poses[i*n_clip:i*n_clip-self.frame_num:-1]

                poses = poses_set
                assert(len(poses)==len(img_paths))

                print(f'Loading {len(img_paths)} {split} images ...')
                for img_path, pose in tqdm(zip(img_paths, poses)):
                    pose_set = []
                    for pose_i in pose:
                        c2w = np.loadtxt(pose_i)[:3]
                        c2w[:, 3] -= self.shift
                        c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
                        pose_set.append(c2w)
                    self.poses.append( pose_set )

                    # pdb.set_trace()
                    if self.preload:
                        # pdb.set_trace()
                        img = read_image(img_path, self.img_wh)
                        mask = read_mask(img_path, self.img_wh)
                        if 'Jade' in self.root_dir or 'Fountain' in self.root_dir:
                            # pdb.set_trace()
                            img = torch.from_numpy(img)
                            # these scenes have black background, changing to white
                            img[torch.all(img<=0.1, dim=-1)] = 1.0

                        self.rays += [img]
                        self.masks += [mask]
                # pdb.set_trace()

            else:
                print("No gt mode")
                # self generated pseudo gt
                #poses = sorted(glob.glob(os.path.join(self.root_dir, 'pre_frame_pseudo_train_pose', '*.txt')))
                # poses = sorted(glob.glob(os.path.join(self.root_dir, 'pose', '0_????-??-0.txt')))
                # "TanksAndTemple"
                poses = sorted(glob.glob(os.path.join(self.root_dir, 'pose', '0_*.txt')))
                # poses = [pose for pose in poses if '00-0.txt' not in pose]
                # pdb.set_trace()
                for pose in tqdm(poses):
                    c2w = np.loadtxt(pose)[:3]
                    c2w[:, 3] -= self.shift
                    c2w[:, 3] /= 2 * self.scale  # to bound the scene inside [-0.5, 0.5]
                    self.poses += [c2w]

                    img = np.zeros((200*200, 3))
                    # print(f"null shape {img.shape}")
                    self.rays += [img]


            if self.preload:
                self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
                self.masks = torch.FloatTensor(np.stack(self.masks))  # (N_images, hw, ?)
                # pdb.set_trace()

        self.poses = torch.FloatTensor(self.poses).reshape(-1, self.frame_num, 3, 4) # (N_images, frame_num, 3, 4)
        # pdb.set_trace()