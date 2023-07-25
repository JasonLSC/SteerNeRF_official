import pdb

from torch.utils.data import Dataset
import numpy as np
import torch

from .color_utils import read_image

class BaseDataset(Dataset):
    """
    Define length and sampling method
    """
    def __init__(self, root_dir, split='train', downsample=1.0):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        if split=='train':
            self.preload = False
        else:
            self.preload = True

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        return len(self.poses)

    def __getitem__(self, idx):
        if self.split.startswith('train') and not hasattr(self, 'super_sampling_factor'): # NeRF_pretrain
            # self.ray_sampling_strategy = 'same_image'
            # training pose is retrieved in train.py
            if self.ray_sampling_strategy == 'all_images': # randomly select images
                img_idxs = np.random.choice(len(self.poses), self.batch_size)
            elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
                img_idxs = np.random.choice(len(self.poses), 1)[0]

            if not self.preload:
                rays = read_image(self.img_paths[img_idxs], self.img_wh)
            else:
                pix_idxs = np.random.choice(self.img_wh[0] * self.img_wh[1], self.batch_size)
                rays = self.rays[img_idxs, pix_idxs]
            # randomly select pixels

            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      'rgb': rays[:, :3]}
            if rays.shape[-1] == 4: # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]

        elif self.split.startswith('train') and hasattr(self, 'super_sampling_factor'): # E2E_joint_training
            img_idxs = np.random.choice(len(self.poses), 1)[0]

            if not self.preload:
                rays = read_image(self.img_paths[img_idxs], self.img_wh)
                rays = torch.FloatTensor(rays)
                copy_rays = rays.clone()

                ### load low res supervision
                # rays_low_set = []
                # for i in range(1):
                #     low_rgb = read_image(self.low_img_paths[img_idxs*self.n_clip+i],
                #                           (self.low_res_w, self.low_res_h))
                #     rays_low_set.append(low_rgb)
                # rays_low = rays_low_set
                # rays_low = torch.FloatTensor(np.stack(rays_low))

            else:
                rays = self.rays[img_idxs]
                copy_rays = rays.clone()
                # rays_low = self.rays_low[img_idxs]


            # patch training
            all_white_flag =True

            while all_white_flag:
            ### get low res patch coord.
                border_h = 0
                border_w = 0
                idx_grid = np.arange(self.low_res_h * self.low_res_w).reshape([self.low_res_h, self.low_res_w])
                h_start, w_start = \
                    np.random.choice(np.arange(border_h, self.low_res_h - self.patch_res_h - border_h), 1)[0], \
                    np.random.choice(np.arange(border_w, self.low_res_w - self.patch_res_w - border_w), 1)[0]

                pix_idxs = idx_grid[h_start:h_start+self.patch_res_h, w_start:w_start+self.patch_res_w].flatten()

                ### get high res patch coord.
                high_res_idx_grid = np.arange(self.high_res_h * self.high_res_w).reshape([self.high_res_h, self.high_res_w])
                high_res_h_start = int(h_start * self.super_sampling_factor)
                high_res_w_start = int(w_start * self.super_sampling_factor)

                high_res_pix_idxs = high_res_idx_grid[high_res_h_start:high_res_h_start + self.patch_high_res_h, high_res_w_start:high_res_w_start + self.patch_high_res_w].flatten()

                #rays = self.rays[img_idxs, high_res_pix_idxs]
                rays = copy_rays[high_res_pix_idxs]
                # rays_low = rays_low[:, pix_idxs]
                if rays.mean() < 0.80:
                    all_white_flag = False
                else:
                    if 'Synthetic' in self.root_dir or 'Blend' in self.root_dir:
                        all_white_flag = False



            poses = self.poses[img_idxs]
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      # 'rgb_low': rays_low[:, :, :3],
                      'rgb': rays[:, :3], # here, rgb should be high res version
                      'pose': poses.unsqueeze(0),
                      'offset': torch.tensor([h_start, w_start]).to(poses.device).unsqueeze(0)}

            sample.update({'K': self.K})
            if hasattr(self, 'low_res_K'):
                sample.update({'low_res_K': self.low_res_K})

            if rays.shape[-1] == 4: # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]

        else: # eval
            sample = {'pose': self.poses[idx].unsqueeze(0), 'img_idxs': idx}
            if hasattr(self,'rays') and len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                masks = self.masks[idx]
                sample['rgb'] = rays[:, :3]
                sample['mask'] = masks
                if rays.shape[1] == 4: # HDR-NeRF data
                    sample['exposure'] = rays[0, 3] # same exposure for all rays
            sample.update({'offset': torch.tensor([0., 0.]).unsqueeze(0)})
            sample.update({'K': self.K})
            if hasattr(self, 'low_res_K'):
                sample.update({'low_res_K': self.low_res_K})


        return sample