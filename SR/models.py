import os
import pdb
from typing import Tuple


import torch
from torch import nn
import torch.nn.functional as F

from timeit import default_timer as timer

from .sr_model_zoo.edsr import EDSR
from .sr_model_zoo.unet import UNet_feat_slim_multiframe

import matplotlib.pyplot as plt


class ZeroUpsampling(nn.Module):
    def __init__(self, channels: int, upsampling_factor: Tuple[int]):
        super().__init__()
        self.channels = channels
        self.upsampling_factor = upsampling_factor
        kernel = torch.zeros((channels, 1, upsampling_factor[1], upsampling_factor[0]), dtype=torch.float32, requires_grad=False)
        kernel[:, 0, 0, 0] = 1
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor):
        return F.conv_transpose2d(x, self.kernel, stride=self.upsampling_factor, groups=self.channels)

class FeatureExtractionNet(nn.Module):
    def __init__(self, in_channels=3):
        super(FeatureExtractionNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, (3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 8, (3, 3), padding=(1, 1))
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, rgbd):
        x = self.relu1(self.conv1(rgbd))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return torch.cat((rgbd, x), dim=1)


class FeatureReweightingNet(nn.Module):
    def __init__(self, in_channels=3, frame_num=3):
        super(FeatureReweightingNet, self).__init__()
        self.conv1 = nn.Conv2d(frame_num*in_channels, 32, (3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, frame_num-1, (3, 3), padding=(1, 1))
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, rgbd):
        x = self.relu1(self.conv1(rgbd))
        x = self.relu2(self.conv2(x))
        x = (torch.tanh(self.conv3(x)) + 1.) # * 5. 
        return x 


DatasetInfoDict = {
    'Synthetic_NeRF':{
        'w':800, 'h':800, 'patch_w':150, 'patch_h':150
    },
    'TanksAndTemple':{
        'w':1920, 'h':1080, 'patch_w':100, 'patch_h':100
    },
    'BlendedMVS':{
        'w':768, 'h':576, 'patch_w':100, 'patch_h':100
    },
}

class NeuralSupersamplingModel(nn.Module):
    def __init__(self, frame_num: int,
                 super_sampling_factor: int = 2,
                 model_type: str = 'nss',  # 'edsr'
                 joint_training: bool = False, 
                 warp_on_training: bool = False,
                 K = None, im_width=200, im_height=200,
                 in_channels=4,
                 feat_training: bool = False,
                 patch_size: int = 150,
                 dataset_type: str = 'Synthetic_NeRF',
                 version: str = 'v2'):

        super().__init__()
        assert frame_num >= 1
        self.frame_num = frame_num
        self.model_type = model_type
        self.joint_training = joint_training

        self.warp_on_training = warp_on_training

        self.feat_training = feat_training

        self.upsampling_factor = (super_sampling_factor, super_sampling_factor)

        self.register_buffer("upsampling_factor_tensor", torch.Tensor([
            self.upsampling_factor[1],
            self.upsampling_factor[0],
        ]).reshape((1, 2, 1, 1)))

        hw_info = DatasetInfoDict[dataset_type]

        self.rgb_output = torch.zeros(1, 3, 6, hw_info['h'], hw_info['w']).to('cuda') # 800, 800 ; 1080, 1920 ; 576, 768
        self.depth_output = torch.zeros(1, 3, 1, hw_info['h'], hw_info['w']).to('cuda')

        self.patch_rgb_output = torch.zeros(1, 3, 6, hw_info['patch_h'], hw_info['patch_w']).to('cuda') # 100 * s
        self.patch_depth_output = torch.zeros(1, 3, 1, hw_info['patch_h'], hw_info['patch_w']).to('cuda')

        # Neural Supersampling
        if 'multiframe' in model_type:
            self.feature_zero_upsampling = ZeroUpsampling(12, self.upsampling_factor)
            self.rgbd_zero_upsampling = ZeroUpsampling(4, self.upsampling_factor)
            self.rgb_zero_upsampling = ZeroUpsampling(3, self.upsampling_factor)
            self.depth_zero_upsampling = ZeroUpsampling(1, self.upsampling_factor)
            self.feature_extraction = FeatureExtractionNet(in_channels=in_channels)
            self.feature_reweighting = FeatureReweightingNet(in_channels=in_channels, frame_num=frame_num)
            self.mask_upsampling = nn.UpsamplingBilinear2d(scale_factor=super_sampling_factor)

        ablation = False
        if ablation: # ablation
            self.feature_zero_upsampling = ZeroUpsampling(12, self.upsampling_factor)
            self.feature_extraction = FeatureExtractionNet()

        self.in_channels = in_channels

        self.model_type=model_type

        print(f"SR model type:{model_type}")

        # EDSR
        if model_type == 'edsr':
            self.model = EDSR(
                in_channels=in_channels*frame_num,
                out_channels=3,
                mid_channels=64,
                num_blocks=8,
                upscale_factor=super_sampling_factor,
                res_scale=1,
            )

        if model_type == 'unet_feature_slim_multiframe':
            self.rgb_upsample = torch.nn.UpsamplingBilinear2d(scale_factor=4)
            self.model = UNet_feat_slim_multiframe(
                in_channels=in_channels,
                upsampling_factor=self.upsampling_factor,
                frame_num=frame_num
            )

        print(self.model)

        # initialize camera ray for warping
        if K is not None:
            self.K = K.view(-1,3,3).to(torch.float32)
            Ki = torch.inverse(K)

            self.im_height = im_height
            self.im_width = im_width

            # low-res ray
            i, j = torch.meshgrid(torch.linspace(0, im_width-1, im_width), torch.linspace(0, im_height-1, im_height))  # pytorch's meshgrid has indexing='ij'
            i = i.t()
            j = j.t()
            self.i = i.cuda()
            self.j = j.cuda()
            K = K.view(3,3)
            ray = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
            self.ray = ray.reshape(1,-1,3).to(torch.float32)

            # high-res ray
            s = super_sampling_factor
            K_highres = K * s 
            K_highres[2,2]=1
            i, j = torch.meshgrid(torch.linspace(0, im_width*s-1, im_width*s), torch.linspace(0, im_height*s-1, im_height*s))  # pytorch's meshgrid has indexing='ij'
            i = i.t()
            j = j.t()
            self.i = i
            self.j = j
            K = K.view(3,3)
            ray = torch.stack([(i-K_highres[0][2])/K_highres[0][0], (j-K_highres[1][2])/K_highres[1][1], torch.ones_like(i)], -1)
            self.ray_highres = ray.reshape(1,-1,3).to(torch.float32)
            self.K_highres = K_highres.view(-1,3,3).to(torch.float32)

        # Initialize meshgrid to save time
        patch_size_h = patch_size*super_sampling_factor
        i, j = torch.meshgrid(torch.linspace(0, patch_size_h-1, patch_size_h), 
                              torch.linspace(0, patch_size_h-1, patch_size_h))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        self.patch_i = i.cuda().unsqueeze(0)
        self.patch_j = j.cuda().unsqueeze(0)
        self.patch_size_h = patch_size_h

        i, j = torch.meshgrid(torch.linspace(0, patch_size-1, patch_size), 
                              torch.linspace(0, patch_size-1, patch_size))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        self.patch_i_low = i.cuda().unsqueeze(0)
        self.patch_j_low = j.cuda().unsqueeze(0)
        self.patch_size_l = patch_size

    def get_K(self, im_height, im_width, offset, high_res=False):
        B = offset.shape[0]
        if high_res:
            K = self.K_highres.view(3,3)
            offset_ = offset * self.upsampling_factor[0]
        else:
            K = self.K.view(3,3)
            offset_ = offset
        K = K.to(offset.device).unsqueeze(0).repeat(B, 1,1)
        # K[:,0,2] = im_width / 2 - offset[:,0] * K[:,0,2] 
        # K[:,1,2] = im_height / 2 - offset[:,1] * K[:,1,2]
        K[:,0,2] -= offset_[:,1]  
        K[:,1,2] -= offset_[:,0] 
        return K

    def get_ray(self, im_height, im_width, offset, high_res=False):
        K = self.get_K(im_height, im_width, offset, high_res)
        # pdb.set_trace()
        B = offset.shape[0]

        if high_res:
            if im_height == self.patch_size_h:
                i = self.patch_i
                j = self.patch_j
            else:
                i = self.i
                j = self.j
        else:
            if im_height == self.patch_size_l:
                i = self.patch_i_low
                j = self.patch_j_low
            else:
                i = self.i_low
                j = self.j_low
        # pdb.set_trace()
        ray = torch.stack([(i-K[:,0,2].reshape(B,1,1))/K[:,0,0].reshape(B,1,1), 
                           (j-K[:,1,2].reshape(B,1,1))/K[:,1,1].reshape(B,1,1), 
                            torch.ones_like(i)], -1)

        return ray.reshape(B,-1,3).to(torch.float32)

    def unproject(self, depth, pose, offset=None, high_res=False):
        torch.cuda.synchronize()
        start = timer()
        ray = self.get_ray(depth.shape[-2], depth.shape[-1], offset, high_res)
        torch.cuda.synchronize()
        end = timer()
        print(f"get ray:{(end - start)*1000} ms")
        # if offset is not None:
        #     import ipdb;ipdb.set_trace()
        # if not high_res:
        #     ray = self.ray.to(depth.device)
        # else:
        #     ray = self.ray_highres.to(depth.device)
        bs = depth.shape[0]
        # pdb.set_trace()
        xyz = depth.reshape(bs,-1,1) * ray 

        # c2w
        xyz = torch.cat((xyz, torch.ones_like(xyz[...,-1:])), -1)
        xyz = (pose @ xyz.transpose(1,2)).transpose(1,2)
        xyz = xyz[...,0:3]

        return xyz

    def vis_pointcloud(self, xyz):
        xyz_ = xyz.detach().cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(xyz_[0,...,0], xyz_[0,...,1], xyz_[0,...,2],'.')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    def project(self, xyz, pose, im_height, im_width, offset, high_res=False):
        # if not high_res:
        #     K = self.K.to(xyz.device)
        # else:
        #     K = self.K_highres.to(xyz.device)
        K = self.get_K(im_height, im_width, offset, high_res)
        bs = xyz.shape[0]
  
        # w2c
        xyz = torch.cat((xyz, torch.ones_like(xyz[...,-1:])), -1)

        pose_inv = torch.clone(pose)
        pose_inv[:,:3,:3] = pose[:,:3,:3].transpose(1,2) 
        pose_inv[:,:3,3:] = -pose[:,:3,:3].transpose(1,2) @ pose[:,:3,3:]
        xyz = (pose_inv @ xyz.transpose(1,2)).transpose(1,2)
        #xyz = (torch.inverse(pose) @ xyz.transpose(1,2)).transpose(1,2)

        xyz = xyz[...,0:3]

        Kt = K.transpose(1,2)
        uv = torch.bmm(xyz, Kt)
        d = uv[:,:,2:3]
  
        # avoid division by zero
        uv = uv[:,:,:2] / (torch.nn.functional.relu(d) + 1e-12)
        return uv, d


    # warp 1,...,T frames to 0th frame
    def warp(self, rgb, depth, pose, mask=None):
        B, I, _, H, W = rgb.shape
        rgb_output = torch.zeros_like(rgb)
        rgb_output[:,0] = rgb[:,0]
        depth_output = torch.zeros_like(depth)
        depth_output[:,0] = depth[:,0]
        for i in range(1,I):
            # project 3D points of frame 0 to [1,...,T] frames
            xyz = self.unproject(depth[:,0], pose[:,0])
            ## DEBUG
            if False:
                xyz2 = self.unproject(depth[:,i], pose[:,i])
                xyz_ = xyz.detach().cpu().numpy()
                xyz2_ = xyz2.detach().cpu().numpy()
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.plot(xyz_[0,...,0], xyz_[0,...,1], xyz_[0,...,2],'r.')
                ax.plot(xyz2_[0,...,0], xyz2_[0,...,1], xyz2_[0,...,2],'b.')
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                plt.show()
            ###
            uv1, d1 = self.project(xyz, pose[:,i])
            # retreive color from other frames
            uv1 = uv1.view(-1, self.im_height, self.im_width, 2).clone()
            uv1[..., 0] = 2 * (uv1[..., 0] / (self.im_width-1) - 0.5)
            uv1[..., 1] = 2 * (uv1[..., 1] / (self.im_height-1) - 0.5)

            rgb10 = torch.nn.functional.grid_sample(rgb[:,i], uv1, padding_mode='border')
            d10 = torch.nn.functional.grid_sample(depth[:,i], uv1, padding_mode='border')
            dis_thres = 0.25
            # TODO: ablation study, mask according to depth inconsistency or not?
            invalid_mask = torch.abs(depth[:,0]-d10)>dis_thres
            d10[invalid_mask]=0
            rgb10[invalid_mask.repeat(1,3,1,1)]=0
            rgb_output[:,i] = rgb10  
            depth_output[:,i] = d10
        return rgb_output, depth_output


    # warp 1,...,T frames to 0th frame
    def warp_zero_upsampling(self, rgb, depth, pose, offset=None, feat=None):
        time_logger = []
        message_info = []
        infer_speedup = True

        torch.cuda.synchronize()
        time_logger.append(timer()) # start

        B, I, C, H, W = rgb.shape
        W_t = W * self.upsampling_factor[0]
        H_t = H * self.upsampling_factor[1]

        if C==3:
            if infer_speedup:  ### assign instead of transpose_conv2d
                rgb_temp_tensor = torch.zeros(B * I, 3, H_t, W_t, device=rgb.device)
                rgb_temp_tensor[:, :, ::self.upsampling_factor[1], ::self.upsampling_factor[0]] = rgb.flatten(0, 1)
                rgb_output = rgb_temp_tensor.reshape(B, I, -1, H_t, W_t)
            else:
                rgb_output = self.rgb_zero_upsampling(rgb.flatten(0,1)).reshape(B,I,-1,H_t,W_t)

            torch.cuda.synchronize()
            time_logger.append(timer()) # rgb upsampling
            message_info.append('rgb upsampling')

            if feat is not None:
                if infer_speedup:  ### assign instead of transpose_conv2d
                    feat_temp_tensor = torch.zeros(B * I, 3, H_t, W_t, device=feat.device)
                    feat_temp_tensor[:, :, ::self.upsampling_factor[1], ::self.upsampling_factor[0]] = feat.flatten(0, 1)
                    feat_output = feat_temp_tensor.reshape(B, I, -1, H_t, W_t)
                else:
                    feat_output = self.rgb_zero_upsampling(feat.flatten(0,1)).reshape(B,I,-1,H_t,W_t)

                rgb_output = torch.cat((rgb_output, feat_output), 2)
                torch.cuda.synchronize()
                time_logger.append(timer()) # feat upsampling + concat
                message_info.append('feat upsampling + concat')
        else:
            rgb_output = self.feature_zero_upsampling(rgb.flatten(0,1)).reshape(B,I,-1,H_t,W_t)

        if infer_speedup: ### assign instead of transpose_conv2d
            upsampled_tensor = torch.zeros(B*I, 1, H_t,W_t, device=depth.device)
            upsampled_tensor[:, :, ::self.upsampling_factor[1], ::self.upsampling_factor[0]] = depth.flatten(0,1)
            depth_output = upsampled_tensor.reshape(B,I,-1,H_t,W_t)
        else:
            depth_output = self.depth_zero_upsampling(depth.flatten(0,1)).reshape(B,I,-1,H_t,W_t)

        torch.cuda.synchronize()
        time_logger.append(timer())  # depth upsampling
        message_info.append('depth upsampling ')

        if self.frame_num==1:
            return rgb_output, depth_output

        for i in range(1,I):
            # project 3D points of frame 0 to [1,...,T] frames
            #di_upsampled = self.depth_zero_upsampling(depth[:,i])
            warping_time_logger = []
            warping_time_message_info = []

            torch.cuda.synchronize()
            warping_time_logger.append(timer())

            di_upsampled = depth_output[:,i]
            xyz = self.unproject(di_upsampled, pose[:,i], offset=offset, high_res=True)

            torch.cuda.synchronize()
            warping_time_logger.append(timer())
            warping_time_message_info.append('unproject')

            t1 = timer()
            # print(f"unprojection inference time: {(t1 - t0) * 1000}ms")
            ## DEBUG
            if False:
                d0_upsampled = self.depth_zero_upsampling(depth[:,0]) 
                xyz2 = self.unproject(d0_upsampled, pose[:,0], offset=offset, high_res=True)
                xyz_ = xyz.detach().cpu().numpy()
                xyz2_ = xyz2.detach().cpu().numpy()
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.plot(xyz_[0,...,0], xyz_[0,...,1], xyz_[0,...,2],'r.')
                ax.plot(xyz2_[0,...,0], xyz2_[0,...,1], xyz2_[0,...,2],'b.')
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                plt.show()
                # import ipdb;ipdb.set_trace()
            ###
            # project to frame 0
            uv0, d0 = self.project(xyz, pose[:,0], H_t, W_t, offset=offset, high_res=True)

            torch.cuda.synchronize()
            warping_time_logger.append(timer())
            warping_time_message_info.append('project')

            uv0 = torch.round(uv0).to(torch.long)
            u_mask = torch.logical_and(uv0[...,0]>=0 , uv0[...,0]<W_t)
            v_mask = torch.logical_and(uv0[...,1]>=0 , uv0[...,1]<H_t)
            uv_mask = torch.logical_and(u_mask, v_mask)
            uv0[...,1] = uv0[...,1].clamp(min=0, max=H_t-1)
            uv0[...,0] = uv0[...,0].clamp(min=0, max=W_t-1)
            for k in range(rgb.shape[0]):
                uv_round = uv0
                rgbi_up = rgb_output[k:k+1, i]
                rgbi_proj = torch.zeros_like(rgbi_up) # B,C,H,W

                rgbi_up = rgbi_up.reshape(*rgbi_up.shape[0:2],-1) # B,C,H,W -> B,C,H*W
                rgbi_up = rgbi_up.permute(0,2,1)
                rgbi_proj[0, :, uv_round[k, uv_mask[k], 1], uv_round[k, uv_mask[k], 0]] = rgbi_up[0, uv_mask[k]].permute(1, 0)

                rgb_output[k:k+1,i] = rgbi_proj

            torch.cuda.synchronize()
            warping_time_logger.append(timer())
            warping_time_message_info.append('get color correspondences')
            for i in range(len(warping_time_logger) - 1):
                print(f'{warping_time_message_info[i]}: {(warping_time_logger[i + 1] - warping_time_logger[i]) * 1e3} ms')



        torch.cuda.synchronize()
        time_logger.append(timer())  # depth-based warping
        message_info.append('depth-based warping ')

        print("====== Runtime on Warp zero upsamling =====")
        for i in range(len(time_logger)-1):
            print(f'{message_info[i]}: {(time_logger[i+1] - time_logger[i])*1e3} ms')

        return rgb_output, depth_output

    # warp 1,...,T frames to 0th frame
    def warp_direct(self, rgb, depth, pose, offset=None, feat=None):
        B, I, C, H, W = rgb.shape
        W_t = W * self.upsampling_factor[0]
        H_t = H * self.upsampling_factor[1]
        if feat is not None:
            rgb = torch.cat((rgb,feat),2)
        # rgb_output = torch.zeros(*rgb.shape[0:3], H_t, W_t).to(rgb.device)
        # depth_output = torch.zeros(*depth.shape[0:3], H_t, W_t).to(rgb.device)
        if W_t == self.im_width:
            rgb_output = self.rgb_output.zero_()
            depth_output = self.depth_output.zero_()
        else:
            rgb_output = self.patch_rgb_output.zero_()
            depth_output = self.patch_depth_output.zero_()
        # pdb.set_trace()
        rgb_output[:,0,...,::4,::4] = rgb[:,0]

        torch.cuda.synchronize()
        start = timer()

        if self.frame_num==1:
            return rgb_output, depth_output

        torch.cuda.synchronize()
        p1 = timer()
        print(f"zero upsampling inference time: {(p1 - start) * 1000}ms")

        for i in range(1,I):
            # project 3D points of frame 0 to [1,...,T] frames
            #di_upsampled = self.depth_zero_upsampling(depth[:,i])
            torch.cuda.synchronize()
            t0 = timer()

            di = depth[:,i]
            xyz = self.unproject(di, pose[:,i], offset=offset, high_res=False)
            if False:
                d0 = depth[:,0] 
                xyz2 = self.unproject(d0, pose[:,0], offset=offset, high_res=False)
                xyz_ = xyz.detach().cpu().numpy()
                xyz2_ = xyz2.detach().cpu().numpy()
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.plot(xyz_[0,...,0], xyz_[0,...,1], xyz_[0,...,2],'r.')
                ax.plot(xyz2_[0,...,0], xyz2_[0,...,1], xyz2_[0,...,2],'b.')
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                plt.show()
                # import ipdb;ipdb.set_trace()

            torch.cuda.synchronize()
            t1 = timer()
            print(f"unprojection inference time: {(t1 - t0) * 1000}ms")
            # project to frame 0
            uv0, d0 = self.project(xyz, pose[:,0], H, W, offset=offset, high_res=False)

            uv0_up = torch.round(uv0*4).to(torch.long)
            uv0 = torch.round(uv0)
            u_mask = torch.logical_and(uv0[...,0]>=0 , uv0[...,0]<W)
            v_mask = torch.logical_and(uv0[...,1]>=0 , uv0[...,1]<H)
            uv_mask = torch.logical_and(u_mask, v_mask)
            u_mask = torch.logical_and(uv0_up[...,0]>=0 , uv0_up[...,0]<W_t)
            v_mask = torch.logical_and(uv0_up[...,1]>=0 , uv0_up[...,1]<H_t)
            uv_mask_up = torch.logical_and(u_mask, v_mask)
            uv_mask = torch.logical_and(uv_mask, uv_mask_up)
            #uv0 = uv0.clamp(min=0, max=H-1)
            uv0_up[...,0] = uv0_up[...,0].clamp(min=0,max=W_t-1)
            uv0_up[...,1] = uv0_up[...,1].clamp(min=0,max=H_t-1)
            torch.cuda.synchronize()
            t2 = timer()
            print(f"projection inference time: {(t2 - t1) * 1000}ms")

            for k in range(rgb.shape[0]):
                uv_round = uv0_up
                rgbi = rgb[k:k+1, i]
                rgbi_proj = rgb_output[k:k+1,i] # B,C,H,W

                rgbi = rgbi.reshape(*rgbi.shape[0:2],-1) # B,C,H,W -> B,C,H*W
                rgbi = rgbi.permute(0,2,1)
                #import ipdb;ipdb.set_trace()
                #rgbi_proj[0, :, uv_round[k, uv_mask[k], 1], uv_round[k, uv_mask[k], 0]] = rgbi[0, uv_mask[k]].permute(1, 0)
                rgbi_proj[0, :, uv_round[k, :, 1], uv_round[k, :, 0]] = rgbi[0].permute(1, 0) * uv_mask.to(rgbi_proj.dtype)
                # import ipdb;ipdb.set_trace()
                # plt.imshow(rgbi_proj[0,0:3].permute(1,2,0).detach().cpu().float().numpy())
                # plt.show()
                rgb_output[k:k+1,i] = rgbi_proj

            torch.cuda.synchronize()
            t2 = timer()
            print(f"round & assign inference time: {(t2 - t1) * 1000}ms")

        return rgb_output, depth_output

    '''
    B: batchsize, I: frame_num

    rgb:    BxIx3xHxW
    depth:  BxIx1xHxW
    pose:   BxIx3x4 or BxIx4x4
    offset: Bx2
    K:      3x3
    feat:   BxIx3xHxW
    '''
    def forward(self, rgb, depth, pose, offset=None, K=None, mask=None, feat=None):
        if not hasattr(self, 'K'):
            assert(K is not None)
            self.K = K.view(-1,3,3).to(torch.float32).cuda()

            # high-res K
            s = self.upsampling_factor[0]
            K_highres = K * s 
            K_highres[2,2]=1
            # pdb.set_trace()
            self.K_highres = K_highres.view(-1,3,3).to(torch.float32)
            # im_height = int(torch.floor(self.K[0,1,2]).item() * 2 * s)
            # im_width = int(torch.floor(self.K[0,0,2]).item() * 2 * s)

            im_height = self.im_height
            im_width =self.im_width

            # pdb.set_trace()
            i, j = torch.meshgrid(torch.linspace(0, im_width-1, im_width), torch.linspace(0, im_height-1, im_height))  # pytorch's meshgrid has indexing='ij'
            i = i.t()
            j = j.t()

            self.i = i.to(self.K.device).unsqueeze(0).repeat(1,1,1)
            self.j = j.to(self.K.device).unsqueeze(0).repeat(1,1,1)

            # pdb.set_trace()
            # im_height = int(torch.floor(self.K[0,1,2]).item() * 2)
            # im_width = int(torch.floor(self.K[0,0,2]).item() * 2)
            im_height = self.im_height // s
            im_width = self.im_width // s
            i, j = torch.meshgrid(torch.linspace(0, im_width-1, im_width), torch.linspace(0, im_height-1, im_height))  # pytorch's meshgrid has indexing='ij'
            i = i.t()
            j = j.t()

            self.i_low = i.to(self.K.device).unsqueeze(0).repeat(1,1,1)
            self.j_low = j.to(self.K.device).unsqueeze(0).repeat(1,1,1)

            # self.rgb_output = torch.zeros(1, 3, 6, self.im_height, self.im_width).to('cuda')
            # self.depth_output = torch.zeros(1, 3, 1, self.im_height, self.im_width).to('cuda')

        # pose: BxIx3x4 -> BxIx4x4
        if pose.shape[-2]==3:
            line = torch.zeros_like(pose[...,0:1,:])
            line[...,-1]=1
            pose = torch.cat((pose, line), -2)

        # do the main job
        if 'multiframe' in self.model_type:
            B, I, _, H, W = rgb.shape
            rgb_up = self.rgb_upsample(rgb[:, 0])

            torch.cuda.synchronize()
            start = timer()
            rgb, depth = self.warp_zero_upsampling(rgb, depth, pose, offset, feat) # rgb.shape = [B, frame_num, 'C', H, W]
            # rgb, depth = self.warp_direct(rgb, depth, pose, offset, feat) # rgb.shape = [B, frame_num, 'C', H, W]
            torch.cuda.synchronize()
            end = timer()

            if feat is not None:
                feat = rgb[:,:,3:6]
            rgb = rgb[:,:,0:3]

            if False:
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(1, 3, figsize=(16, 9))
                axs[0].imshow(rgb[0, 0].clamp(min=0., max=1.).permute(1, 2, 0).detach().float().cpu().numpy())
                axs[1].imshow(rgb[0, 1].clamp(min=0., max=1.).permute(1, 2, 0).detach().float().cpu().numpy())
                axs[2].imshow(rgb[0, 2].clamp(min=0., max=1.).permute(1, 2, 0).detach().float().cpu().numpy())
                plt.show()
                fig, axs = plt.subplots(1, 3, figsize=(16, 9))
                axs[0].imshow(feat[0, 0].clamp(min=0., max=1.).permute(1, 2, 0).detach().float().cpu().numpy())
                axs[1].imshow(feat[0, 1].clamp(min=0., max=1.).permute(1, 2, 0).detach().float().cpu().numpy())
                axs[2].imshow(feat[0, 2].clamp(min=0., max=1.).permute(1, 2, 0).detach().float().cpu().numpy())
                plt.show()

            rgb = rgb.reshape([B, -1, *rgb.shape[-2:]])

            if feat is not None:
                feat = feat.reshape([B, -1, *rgb.shape[-2:]])
                rgb = torch.cat((rgb,feat), dim=1)

            torch.cuda.synchronize()
            network_start = timer()
            reconstructed = self.model(rgb, rgb_up)

            torch.cuda.synchronize()
            elapsed_time = timer() - network_start

        return reconstructed, elapsed_time


class trt_NeuralSupersamplingModel(nn.Module):
    def __init__(self, frame_num: int,
                 super_sampling_factor: int = 2,
                 model_type: str = 'nss',  # 'edsr'
                 joint_training: bool = False,
                 warp_on_training: bool = False,
                 K=None, im_width=200, im_height=200,
                 in_channels=4,
                 feat_training: bool = False,
                 patch_size: int = 150,
                 dataset_type: str = 'Synthetic_NeRF',
                 TRT_engine_file: str = ''):

        super().__init__()
        assert frame_num >= 1
        self.frame_num = frame_num
        self.model_type = model_type
        self.joint_training = joint_training

        self.warp_on_training = warp_on_training

        self.feat_training = feat_training

        self.upsampling_factor = (super_sampling_factor, super_sampling_factor)

        self.register_buffer("upsampling_factor_tensor", torch.Tensor([
            self.upsampling_factor[1],
            self.upsampling_factor[0],
        ]).reshape((1, 2, 1, 1)))

        import tensorrt as trt
        from utils import TRTModule
        engine_path = TRT_engine_file
        logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        trt_model = TRTModule(engine, ['rgb', 'rgb_up'], ['output'])
        self.trt_model = trt_model

        hw_info = DatasetInfoDict[dataset_type]

        self.rgb_output = torch.zeros(1, frame_num, 6, hw_info['h'], hw_info['w']).to('cuda')
        self.depth_output = torch.zeros(1, frame_num, 1, hw_info['h'], hw_info['w']).to('cuda')

        self.patch_rgb_output = torch.zeros(1, frame_num, 6, hw_info['patch_h'], hw_info['patch_w']).to('cuda')
        self.patch_depth_output = torch.zeros(1, frame_num, 1, hw_info['patch_h'], hw_info['patch_w']).to('cuda')

        # Neural Supersampling
        if 'nss' in model_type or 'multiframe' in model_type:
            self.feature_zero_upsampling = ZeroUpsampling(12, self.upsampling_factor)
            self.rgbd_zero_upsampling = ZeroUpsampling(4, self.upsampling_factor)
            self.rgb_zero_upsampling = ZeroUpsampling(3, self.upsampling_factor)
            self.depth_zero_upsampling = ZeroUpsampling(1, self.upsampling_factor)
            self.feature_extraction = FeatureExtractionNet(in_channels=in_channels)
            self.feature_reweighting = FeatureReweightingNet(in_channels=in_channels, frame_num=frame_num)
            self.mask_upsampling = nn.UpsamplingBilinear2d(scale_factor=super_sampling_factor)

        self.in_channels = in_channels
        self.model_type = model_type

        # EDSR
        if model_type == 'edsr':
            self.model = EDSR(
                in_channels=in_channels * frame_num,
                out_channels=3,
                mid_channels=64,
                num_blocks=8,
                upscale_factor=super_sampling_factor,
                res_scale=1,
            )

        if model_type == 'unet_feature_slim_multiframe':
            self.rgb_upsample = torch.nn.UpsamplingBilinear2d(scale_factor=4)
            self.model = UNet_feat_slim_multiframe(
                in_channels=in_channels,
                upsampling_factor=self.upsampling_factor,
                frame_num=frame_num
            )

        # initialize camera ray for warping
        if K is not None:
            self.K = K.view(-1, 3, 3).to(torch.float32)
            Ki = torch.inverse(K)

            self.im_height = im_height
            self.im_width = im_width

            # low-res ray
            i, j = torch.meshgrid(torch.linspace(0, im_width - 1, im_width),
                                  torch.linspace(0, im_height - 1, im_height))  # pytorch's meshgrid has indexing='ij'
            i = i.t()
            j = j.t()
            self.i = i.cuda()
            self.j = j.cuda()
            K = K.view(3, 3)
            ray = torch.stack([(i - K[0][2]) / K[0][0], (j - K[1][2]) / K[1][1], torch.ones_like(i)], -1)
            self.ray = ray.reshape(1, -1, 3).to(torch.float32)

            # high-res ray
            s = super_sampling_factor
            K_highres = K * s
            K_highres[2, 2] = 1
            i, j = torch.meshgrid(torch.linspace(0, im_width * s - 1, im_width * s),
                                  torch.linspace(0, im_height * s - 1,
                                                 im_height * s))  # pytorch's meshgrid has indexing='ij'
            i = i.t()
            j = j.t()
            self.i = i
            self.j = j
            K = K.view(3, 3)
            ray = torch.stack(
                [(i - K_highres[0][2]) / K_highres[0][0], (j - K_highres[1][2]) / K_highres[1][1], torch.ones_like(i)],
                -1)
            self.ray_highres = ray.reshape(1, -1, 3).to(torch.float32)
            self.K_highres = K_highres.view(-1, 3, 3).to(torch.float32)

        # Initialize meshgrid to save time
        patch_size_h = patch_size * super_sampling_factor
        i, j = torch.meshgrid(torch.linspace(0, patch_size_h - 1, patch_size_h),
                              torch.linspace(0, patch_size_h - 1, patch_size_h))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        self.patch_i = i.cuda().unsqueeze(0)
        self.patch_j = j.cuda().unsqueeze(0)
        self.patch_size_h = patch_size_h

        i, j = torch.meshgrid(torch.linspace(0, patch_size - 1, patch_size),
                              torch.linspace(0, patch_size - 1, patch_size))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        self.patch_i_low = i.cuda().unsqueeze(0)
        self.patch_j_low = j.cuda().unsqueeze(0)
        self.patch_size_l = patch_size

    def get_K(self, im_height, im_width, offset, high_res=False):
        B = offset.shape[0]
        if high_res:
            K = self.K_highres.view(3, 3)
            offset_ = offset * self.upsampling_factor[0]
        else:
            K = self.K.view(3, 3)
            offset_ = offset
        K = K.to(offset.device).unsqueeze(0).repeat(B, 1, 1)
        # K[:,0,2] = im_width / 2 - offset[:,0] * K[:,0,2]
        # K[:,1,2] = im_height / 2 - offset[:,1] * K[:,1,2]
        K[:, 0, 2] -= offset_[:, 1]
        K[:, 1, 2] -= offset_[:, 0]
        return K

    def get_ray(self, im_height, im_width, offset, high_res=False):
        K = self.get_K(im_height, im_width, offset, high_res)
        # pdb.set_trace()
        B = offset.shape[0]

        if high_res:
            if im_height == self.patch_size_h:
                i = self.patch_i
                j = self.patch_j
            else:
                i = self.i
                j = self.j
        else:
            if im_height == self.patch_size_l:
                i = self.patch_i_low
                j = self.patch_j_low
            else:
                i = self.i_low
                j = self.j_low
        # pdb.set_trace()
        ray = torch.stack([(i - K[:, 0, 2].reshape(B, 1, 1)) / K[:, 0, 0].reshape(B, 1, 1),
                           (j - K[:, 1, 2].reshape(B, 1, 1)) / K[:, 1, 1].reshape(B, 1, 1),
                           torch.ones_like(i)], -1)

        return ray.reshape(B, -1, 3).to(torch.float32)

    def unproject(self, depth, pose, offset=None, high_res=False):
        # torch.cuda.synchronize()
        # start = timer()
        ray = self.get_ray(depth.shape[-2], depth.shape[-1], offset, high_res)
        # torch.cuda.synchronize()
        # end = timer()
        # print(f"get ray:{(end - start)*1000}")
        # if offset is not None:
        #     import ipdb;ipdb.set_trace()
        # if not high_res:
        #     ray = self.ray.to(depth.device)
        # else:
        #     ray = self.ray_highres.to(depth.device)
        bs = depth.shape[0]
        # pdb.set_trace()
        xyz = depth.reshape(bs, -1, 1) * ray

        # c2w
        xyz = torch.cat((xyz, torch.ones_like(xyz[..., -1:])), -1)
        xyz = (pose @ xyz.transpose(1, 2)).transpose(1, 2)
        xyz = xyz[..., 0:3]

        return xyz

    def vis_pointcloud(self, xyz):
        xyz_ = xyz.detach().cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(xyz_[0, ..., 0], xyz_[0, ..., 1], xyz_[0, ..., 2], '.')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    def project(self, xyz, pose, im_height, im_width, offset, high_res=False):
        # if not high_res:
        #     K = self.K.to(xyz.device)
        # else:
        #     K = self.K_highres.to(xyz.device)
        K = self.get_K(im_height, im_width, offset, high_res)
        bs = xyz.shape[0]

        # w2c
        xyz = torch.cat((xyz, torch.ones_like(xyz[..., -1:])), -1)

        pose_inv = torch.clone(pose)
        pose_inv[:, :3, :3] = pose[:, :3, :3].transpose(1, 2)
        pose_inv[:, :3, 3:] = -pose[:, :3, :3].transpose(1, 2) @ pose[:, :3, 3:]
        xyz = (pose_inv @ xyz.transpose(1, 2)).transpose(1, 2)
        # xyz = (torch.inverse(pose) @ xyz.transpose(1,2)).transpose(1,2)

        xyz = xyz[..., 0:3]

        Kt = K.transpose(1, 2)
        uv = torch.bmm(xyz, Kt)

        d = uv[:, :, 2:3]

        # avoid division by zero
        uv = uv[:, :, :2] / (torch.nn.functional.relu(d) + 1e-12)
        return uv, d

    # warp 1,...,T frames to 0th frame
    def warp(self, rgb, depth, pose, mask=None):
        B, I, _, H, W = rgb.shape
        rgb_output = torch.zeros_like(rgb)
        rgb_output[:, 0] = rgb[:, 0]
        depth_output = torch.zeros_like(depth)
        depth_output[:, 0] = depth[:, 0]
        for i in range(1, I):
            # project 3D points of frame 0 to [1,...,T] frames
            xyz = self.unproject(depth[:, 0], pose[:, 0])
            ## DEBUG
            if False:
                xyz2 = self.unproject(depth[:, i], pose[:, i])
                xyz_ = xyz.detach().cpu().numpy()
                xyz2_ = xyz2.detach().cpu().numpy()
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.plot(xyz_[0, ..., 0], xyz_[0, ..., 1], xyz_[0, ..., 2], 'r.')
                ax.plot(xyz2_[0, ..., 0], xyz2_[0, ..., 1], xyz2_[0, ..., 2], 'b.')
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                plt.show()
            ###
            uv1, d1 = self.project(xyz, pose[:, i])
            # retreive color from other frames
            uv1 = uv1.view(-1, self.im_height, self.im_width, 2).clone()
            uv1[..., 0] = 2 * (uv1[..., 0] / (self.im_width - 1) - 0.5)
            uv1[..., 1] = 2 * (uv1[..., 1] / (self.im_height - 1) - 0.5)

            rgb10 = torch.nn.functional.grid_sample(rgb[:, i], uv1, padding_mode='border')
            d10 = torch.nn.functional.grid_sample(depth[:, i], uv1, padding_mode='border')
            dis_thres = 0.25
            # TODO: ablation study, mask according to depth inconsistency or not?
            invalid_mask = torch.abs(depth[:, 0] - d10) > dis_thres
            d10[invalid_mask] = 0
            rgb10[invalid_mask.repeat(1, 3, 1, 1)] = 0
            rgb_output[:, i] = rgb10
            depth_output[:, i] = d10
        return rgb_output, depth_output

    # warp 1,...,T frames to 0th frame
    def warp_zero_upsampling(self, rgb, depth, pose, offset=None, feat=None):
        time_logger = []
        message_info = []
        infer_speedup = True

        B, I, C, H, W = rgb.shape
        W_t = W * self.upsampling_factor[0]
        H_t = H * self.upsampling_factor[1]

        torch.cuda.synchronize()
        time_logger.append(timer()) # start

        if C == 3:
            rgb_output = self.rgb_zero_upsampling(rgb.flatten(0, 1)).reshape(B, I, -1, H_t, W_t)
            if feat is not None:
                feat_output = self.rgb_zero_upsampling(feat.flatten(0, 1)).reshape(B, I, -1, H_t, W_t)
                rgb_output = torch.cat((rgb_output, feat_output), 2)
        else:
            rgb_output = self.feature_zero_upsampling(rgb.flatten(0, 1)).reshape(B, I, -1, H_t, W_t)
        # depth_output = torch.zeros(*depth.shape[0:3], H_t, W_t).to(depth.devicew)
        # depth_output[:,0] = self.depth_zero_upsampling(depth[:,0])
        depth_output = self.depth_zero_upsampling(depth.flatten(0, 1)).reshape(B, I, -1, H_t, W_t)
        if self.frame_num == 1:
            return rgb_output, depth_output

        torch.cuda.synchronize()
        p1 = timer()
        # print(f"zero upsampling inference time: {(p1 - start) * 1000}ms")

        for i in range(1, I):
            # project 3D points of frame 0 to [1,...,T] frames
            # di_upsampled = self.depth_zero_upsampling(depth[:,i])
            torch.cuda.synchronize()
            t0 = timer()
            # pdb.set_trace()
            di_upsampled = depth_output[:, i]
            xyz = self.unproject(di_upsampled, pose[:, i], offset=offset, high_res=True)

            torch.cuda.synchronize()
            t1 = timer()
            # print(f"unprojection inference time: {(t1 - t0) * 1000}ms")
            ## DEBUG
            if False:
                d0_upsampled = self.depth_zero_upsampling(depth[:, 0])
                xyz2 = self.unproject(d0_upsampled, pose[:, 0], offset=offset, high_res=True)
                xyz_ = xyz.detach().cpu().numpy()
                xyz2_ = xyz2.detach().cpu().numpy()
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.plot(xyz_[0, ..., 0], xyz_[0, ..., 1], xyz_[0, ..., 2], 'r.')
                ax.plot(xyz2_[0, ..., 0], xyz2_[0, ..., 1], xyz2_[0, ..., 2], 'b.')
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                plt.show()
                # import ipdb;ipdb.set_trace()
            ###
            # project to frame 0
            uv0, d0 = self.project(xyz, pose[:, 0], H_t, W_t, offset=offset, high_res=True)

            torch.cuda.synchronize()
            t2 = timer()
            # print(f"projection inference time: {(t2 - t1) * 1000}ms")

            uv0 = torch.round(uv0).to(torch.long)
            u_mask = torch.logical_and(uv0[..., 0] >= 0, uv0[..., 0] < W_t)
            v_mask = torch.logical_and(uv0[..., 1] >= 0, uv0[..., 1] < H_t)
            uv_mask = torch.logical_and(u_mask, v_mask)
            uv0[..., 1] = uv0[..., 1].clamp(min=0, max=H_t - 1)
            uv0[..., 0] = uv0[..., 0].clamp(min=0, max=W_t - 1)
            for k in range(rgb.shape[0]):
                uv_round = uv0
                rgbi_up = rgb_output[k:k + 1, i]
                rgbi_proj = torch.zeros_like(rgbi_up)  # B,C,H,W
                # rgbi_proj =

                rgbi_up = rgbi_up.reshape(*rgbi_up.shape[0:2], -1)  # B,C,H,W -> B,C,H*W
                rgbi_up = rgbi_up.permute(0, 2, 1)
                rgbi_proj[0, :, uv_round[k, uv_mask[k], 1], uv_round[k, uv_mask[k], 0]] = rgbi_up[
                    0, uv_mask[k]].permute(1, 0)
                # import ipdb;ipdb.set_trace()
                # plt.imshow(rgbi_proj[0,0:3].permute(1,2,0).detach().cpu().float().numpy())
                # plt.show()
                rgb_output[k:k + 1, i] = rgbi_proj

            torch.cuda.synchronize()
            t2 = timer()
            # print(f"round & assign inference time: {(t2 - t1) * 1000}ms")

            # uv1 = uv1.view(-1, H_t, W_t, 2).clone()
            # uv1[..., 0] = 2 * (uv1[..., 0] / (W_t-1) - 0.5)
            # uv1[..., 1] = 2 * (uv1[..., 1] / (H_t-1) - 0.5)
            # rgb10 = torch.nn.functional.grid_sample(rgb[:,i], uv1, padding_mode='border')
            # d10 = torch.nn.functional.grid_sample(depth[:,i], uv1, padding_mode='border')
            # import ipdb;ipdb.set_trace()
            # dis_thres = 0.25
            # # TODO: ablation study, mask according to depth inconsistency or not?
            # invalid_mask = torch.abs(depth[:,0]-d10)>dis_thres
            # d10[invalid_mask]=0
            # rgb10[invalid_mask.repeat(1,3,1,1)]=0
            # rgb_output[:,i] = rgb10
            # depth_output[:,i] = d10

        # torch.cuda.synchronize()
        # p2 = timer()
        # print(f"warping(inside) inference time: {(p2 - p1) * 1000}ms")
        return rgb_output, depth_output

    # warp 1,...,T frames to 0th frame
    def warp_direct(self, rgb, depth, pose, offset=None, feat=None):
        time_logger = []
        message_info = []
        infer_speedup = True

        torch.cuda.synchronize()
        time_logger.append(timer()) # start

        B, I, C, H, W = rgb.shape
        W_t = W * self.upsampling_factor[0]
        H_t = H * self.upsampling_factor[1]
        if feat is not None:
            rgb = torch.cat((rgb, feat), 2)

        if W_t == self.im_width:
            rgb_output = self.rgb_output.zero_()
            depth_output = self.depth_output.zero_()
        else:
            rgb_output = self.patch_rgb_output.zero_()
            depth_output = self.patch_depth_output.zero_()

        rgb_output[:, 0, ..., ::4, ::4] = rgb[:, 0]

        torch.cuda.synchronize()
        time_logger.append(timer())  # rgb upsampling
        message_info.append('current frame rgb upsampling')

        if self.frame_num == 1:
            return rgb_output, depth_output

        for i in range(1, I):
            # project 3D points of frame 0 to [1,...,T] frames
            # di_upsampled = self.depth_zero_upsampling(depth[:,i])
            di = depth[:, i]
            xyz = self.unproject(di, pose[:, i], offset=offset, high_res=False)
            if False:
                d0 = depth[:, 0]
                xyz2 = self.unproject(d0, pose[:, 0], offset=offset, high_res=False)
                xyz_ = xyz.detach().cpu().numpy()
                xyz2_ = xyz2.detach().cpu().numpy()
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.plot(xyz_[0, ..., 0], xyz_[0, ..., 1], xyz_[0, ..., 2], 'r.')
                ax.plot(xyz2_[0, ..., 0], xyz2_[0, ..., 1], xyz2_[0, ..., 2], 'b.')
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                plt.show()
                # import ipdb;ipdb.set_trace()

            torch.cuda.synchronize()
            time_logger.append(timer())  # unproject i th frame
            message_info.append(f'unproject feature of No.{i} frame')

            # project to frame 0
            uv0, d0 = self.project(xyz, pose[:, 0], H, W, offset=offset, high_res=False)

            torch.cuda.synchronize()
            time_logger.append(timer())  # project i th frame
            message_info.append(f'project point cloud of No.{i} frame')

            uv0_up = torch.round(uv0 * 4).to(torch.long)
            uv0 = torch.round(uv0)
            u_mask = torch.logical_and(uv0[..., 0] >= 0, uv0[..., 0] < W)
            v_mask = torch.logical_and(uv0[..., 1] >= 0, uv0[..., 1] < H)
            uv_mask = torch.logical_and(u_mask, v_mask)
            u_mask = torch.logical_and(uv0_up[..., 0] >= 0, uv0_up[..., 0] < W_t)
            v_mask = torch.logical_and(uv0_up[..., 1] >= 0, uv0_up[..., 1] < H_t)
            uv_mask_up = torch.logical_and(u_mask, v_mask)
            uv_mask = torch.logical_and(uv_mask, uv_mask_up)
            # uv0 = uv0.clamp(min=0, max=H-1)
            uv0_up[..., 0] = uv0_up[..., 0].clamp(min=0, max=W_t - 1)
            uv0_up[..., 1] = uv0_up[..., 1].clamp(min=0, max=H_t - 1)

            torch.cuda.synchronize()
            time_logger.append(timer())  # get warped coord.
            message_info.append(f'get warped coord. of No.{i} frame')

            for k in range(rgb.shape[0]):
                uv_round = uv0_up
                rgbi = rgb[k:k + 1, i]
                rgbi_proj = rgb_output[k:k + 1, i]  # B,C,H,W

                rgbi = rgbi.reshape(*rgbi.shape[0:2], -1)  # B,C,H,W -> B,C,H*W
                rgbi = rgbi.permute(0, 2, 1)
                # import ipdb;ipdb.set_trace()
                # rgbi_proj[0, :, uv_round[k, uv_mask[k], 1], uv_round[k, uv_mask[k], 0]] = rgbi[0, uv_mask[k]].permute(1, 0)
                rgbi_proj[0, :, uv_round[k, :, 1], uv_round[k, :, 0]] = rgbi[0].permute(1, 0) * uv_mask.to(
                    rgbi_proj.dtype)
                # import ipdb;ipdb.set_trace()
                # plt.imshow(rgbi_proj[0,0:3].permute(1,2,0).detach().cpu().float().numpy())
                # plt.show()
                rgb_output[k:k + 1, i] = rgbi_proj

            torch.cuda.synchronize()
            time_logger.append(timer())  # get warped coord.
            message_info.append(f'copy feature from No.{i} frame')

            # uv1 = uv1.view(-1, H_t, W_t, 2).clone()
            # uv1[..., 0] = 2 * (uv1[..., 0] / (W_t-1) - 0.5)
            # uv1[..., 1] = 2 * (uv1[..., 1] / (H_t-1) - 0.5)
            # rgb10 = torch.nn.functional.grid_sample(rgb[:,i], uv1, padding_mode='border')
            # d10 = torch.nn.functional.grid_sample(depth[:,i], uv1, padding_mode='border')
            # import ipdb;ipdb.set_trace()
            # dis_thres = 0.25
            # # TODO: ablation study, mask according to depth inconsistency or not?
            # invalid_mask = torch.abs(depth[:,0]-d10)>dis_thres
            # d10[invalid_mask]=0
            # rgb10[invalid_mask.repeat(1,3,1,1)]=0
            # rgb_output[:,i] = rgb10
            # depth_output[:,i] = d10

        for i in range(len(time_logger)-1):
            print(f'{message_info[i]}: {(time_logger[i+1] - time_logger[i])*1e3} ms')

        return rgb_output, depth_output

    '''
    B: batchsize, I: frame_num

    rgb:    BxIx3xHxW
    depth:  BxIx1xHxW
    pose:   BxIx3x4 or BxIx4x4
    offset: Bx2
    K:      3x3
    feat:   BxIx3xHxW
    '''

    def forward(self, rgb, depth, pose, offset=None, K=None, mask=None, feat=None):
        if not hasattr(self, 'K'):
            assert (K is not None)
            self.K = K.view(-1, 3, 3).to(torch.float32).cuda()

            # high-res K
            s = self.upsampling_factor[0]
            K_highres = K * s
            K_highres[2, 2] = 1
            # pdb.set_trace()
            self.K_highres = K_highres.view(-1, 3, 3).to(torch.float32)
            # im_height = int(torch.floor(self.K[0,1,2]).item() * 2 * s)
            # im_width = int(torch.floor(self.K[0,0,2]).item() * 2 * s)

            im_height = self.im_height
            im_width = self.im_width

            # pdb.set_trace()
            i, j = torch.meshgrid(torch.linspace(0, im_width - 1, im_width),
                                  torch.linspace(0, im_height - 1, im_height))  # pytorch's meshgrid has indexing='ij'
            i = i.t()
            j = j.t()

            self.i = i.to(self.K.device).unsqueeze(0).repeat(1, 1, 1)
            self.j = j.to(self.K.device).unsqueeze(0).repeat(1, 1, 1)

            # pdb.set_trace()
            # im_height = int(torch.floor(self.K[0,1,2]).item() * 2)
            # im_width = int(torch.floor(self.K[0,0,2]).item() * 2)
            im_height = self.im_height // s
            im_width = self.im_width // s
            i, j = torch.meshgrid(torch.linspace(0, im_width - 1, im_width),
                                  torch.linspace(0, im_height - 1, im_height))  # pytorch's meshgrid has indexing='ij'
            i = i.t()
            j = j.t()

            self.i_low = i.to(self.K.device).unsqueeze(0).repeat(1, 1, 1)
            self.j_low = j.to(self.K.device).unsqueeze(0).repeat(1, 1, 1)

            # self.rgb_output = torch.zeros(1, 3, 6, self.im_height, self.im_width).to('cuda')
            # self.depth_output = torch.zeros(1, 3, 1, self.im_height, self.im_width).to('cuda')

        # pose: BxIx3x4 -> BxIx4x4
        if pose.shape[-2] == 3:
            line = torch.zeros_like(pose[..., 0:1, :])
            line[..., -1] = 1
            pose = torch.cat((pose, line), -2)

        # do the main job
        if self.model_type == 'nss_siggraph':
            rgb_up = self.rgb_upsample(rgb[:, 0])
            # B, I, _, H, W = rgb.shape
            x = torch.cat((rgb, depth / 5.), dim=2)
            # feature extraction network
            B, I, _, H, W = x.shape
            x = x.reshape(B * I, -1, H, W)
            x_feat = self.feature_extraction(x)
            # feature upsampling & warping
            x_feat = x_feat.reshape(B, I, -1, H, W)
            x_feat, depth = self.warp_zero_upsampling(x_feat, depth, pose, offset)
            H = H * self.upsampling_factor[0]
            W = W * self.upsampling_factor[1]
            # feature reweighting
            if self.frame_num > 1:
                rgbd_up = x_feat[:, :, 0:4]
                rgbd_up = rgbd_up.reshape(B, -1, H, W)
                x_weight = self.feature_reweighting(rgbd_up)
                x_weight = torch.cat((torch.ones_like(x_weight[:, 0:1]), x_weight), 1)
                x_feat = x_feat.reshape(B, I, -1, H, W)
                x_weight = x_weight.reshape(B, I, 1, H, W)
                x_feat = x_feat * x_weight
            # reconstruction
            x_feat = x_feat.reshape(B, -1, H, W)
            reconstructed = self.model(x_feat)
            reconstructed = reconstructed + rgb_up
        else:
            if self.model_type == 'nss':
                rgb, depth = self.warp_zero_upsampling(rgb, depth, pose, offset)

        # sli's temp. implementation(need to integrate with yliao's pipeline)
        # rgbd
        if not 'multiframe' in self.model_type:
            if self.in_channels == 4:
                # yliao 's implementation
                rgbd = torch.cat((rgb, depth / 5.), dim=-3)
                # rgbd = torch.cat((rgb, depth), dim=-3) # [B, frame_num, 'C', H, W]
                reconstructed = self.model(rgbd)
            else:
                # rgb + feat
                if feat is not None and 'feat' in self.model_type:
                    rgb_feat = torch.cat((rgb, feat), dim=-3)

                    # torch.cuda.synchronize()
                    # start = timer()

                    reconstructed = self.model(rgb_feat)

                    # torch.cuda.synchronize()
                    # print(f"rec net inference time: {(timer() - start)*1000}ms")
                else:
                    # pdb.set_trace()
                    reconstructed = self.model(rgb)

        if 'multiframe' in self.model_type:

            B, I, _, H, W = rgb.shape
            rgb_up = self.rgb_upsample(rgb[:, 0])

            torch.cuda.synchronize()
            start = timer()
            rgb, depth = self.warp_direct(rgb, depth, pose, offset, feat)  # rgb.shape = [B, frame_num, 'C', H, W]
            # rgb, depth = self.warp_direct(rgb, depth, pose, offset, feat) # rgb.shape = [B, frame_num, 'C', H, W]
            torch.cuda.synchronize()
            end = timer()

            pdb.set_trace() # at this moment, rgb: [1,3,6,800,800]
            if feat is not None:
                feat = rgb[:, :, 3:6]
            rgb = rgb[:, :, 0:3]

            if False:
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(1, 3, figsize=(16, 9))
                axs[0].imshow(rgb[0, 0].clamp(min=0., max=1.).permute(1, 2, 0).detach().float().cpu().numpy())
                axs[1].imshow(rgb[0, 1].clamp(min=0., max=1.).permute(1, 2, 0).detach().float().cpu().numpy())
                axs[2].imshow(rgb[0, 2].clamp(min=0., max=1.).permute(1, 2, 0).detach().float().cpu().numpy())
                plt.show()
                fig, axs = plt.subplots(1, 3, figsize=(16, 9))
                axs[0].imshow(feat[0, 0].clamp(min=0., max=1.).permute(1, 2, 0).detach().float().cpu().numpy())
                axs[1].imshow(feat[0, 1].clamp(min=0., max=1.).permute(1, 2, 0).detach().float().cpu().numpy())
                axs[2].imshow(feat[0, 2].clamp(min=0., max=1.).permute(1, 2, 0).detach().float().cpu().numpy())
                plt.show()

            rgb = rgb.reshape([B, -1, *rgb.shape[-2:]])
            if feat is not None:
                feat = feat.reshape([B, -1, *rgb.shape[-2:]])
                rgb = torch.cat((rgb, feat), dim=1)

            torch.cuda.synchronize()
            network_start = timer()
            reconstructed = self.trt_model(rgb, rgb_up)

            reconstructed = reconstructed[0:1, ...]

            torch.cuda.synchronize()
            network_end = timer() - network_start

        if mask is not None:
            up_mask = self.mask_upsampling(mask)
            tmp = torch.zeros_like(up_mask)
            tmp[up_mask > 0.1] = 1
            up_source_mask = tmp
            up_source_mask = up_source_mask.detach()

            reconstructed = reconstructed * up_source_mask + (1 - up_source_mask)

        return reconstructed, network_end