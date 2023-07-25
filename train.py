import pdb

import torch
from torch import nn
from opt import get_opts
from timeit import default_timer as timer
import os
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP
from models.rendering import render, MAX_SAMPLES
from models.project import Projection
from SR import models as SRmodels

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, ChainedScheduler, ConstantLR, SequentialLR
from losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# utils
from utils import slim_ckpt, load_ckpt

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

import warnings; warnings.filterwarnings("ignore")


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
    return depth_img

def depthScaling(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    return depth


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.automatic_optimization = True

        self.save_hyperparameters(hparams)

        self.training_stage = hparams.training_stage

        self.warmup_steps = 256
        self.update_interval = 16

        self.loss = NeRFLoss(lambda_distortion=self.hparams.distortion_loss_w)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        self.model = NGP(scale=self.hparams.scale, rgb_act=rgb_act, T=self.hparams.log2_T)
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))

        if self.hparams.super_sampling:
            # load NeRF model parameters
            if not self.hparams.val_only: # start joint training
                ngp_ckpt_name = f'ckpts/{self.hparams.dataset_name}/{hparams.exp_name}/NeRF_Pretrain/epoch=19_slim_feat.ckpt'
                # ngp_ckpt_name = f'ckpts/{self.hparams.dataset_name}/{hparams.exp_name}/epoch=19_slim_feat.ckpt'
                load_ckpt(self.model, ngp_ckpt_name)
                print(f"ckpt:", ngp_ckpt_name)
                # pdb.set_trace()

            if 'Synthetic' in hparams.root_dir:
                dataset_type = 'Synthetic_NeRF'
            elif 'Tanks' in hparams.root_dir:
                dataset_type = 'TanksAndTemple'
            elif 'Blend' in hparams.root_dir:
                dataset_type = 'BlendedMVS'

            if not self.hparams.TRT_enable:
                print("Using Pytorch U-Net for inference")
                self.SRmodel = SRmodels.NeuralSupersamplingModel(frame_num=self.hparams.frame_num,
                                                                model_type=self.hparams.sr_model_type,
                                                                super_sampling_factor=self.hparams.super_sampling_factor,
                                                                joint_training=self.hparams.super_sampling,
                                                                in_channels=6 if self.hparams.feature_training else 3,
                                                                         # no feat:3; with feat:3+3
                                                                feat_training=self.hparams.feature_training,
                                                                patch_size=self.hparams.patch_size,
                                                                dataset_type=dataset_type)
            else:
                # import tensorrt as trt
                # from utils import TRTModule
                print("Using TensorRT-optimized U-Net for inference")
                self.SRmodel = SRmodels.trt_NeuralSupersamplingModel(frame_num=self.hparams.frame_num,
                                                                     model_type=self.hparams.sr_model_type,
                                                                     super_sampling_factor=self.hparams.super_sampling_factor,
                                                                     joint_training=self.hparams.super_sampling,
                                                                     in_channels=6 if self.hparams.feature_training else 3,
                                                                     # no feat:3; with feat:3+3
                                                                     feat_training=self.hparams.feature_training,
                                                                     patch_size=self.hparams.patch_size,
                                                                     dataset_type=dataset_type,
                                                                     TRT_engine_file=self.hparams.TRT_engine_file)

            ### DEBUG: skip loading model
            if False:
                self.SRmodel.load_state_dict(state_dict_updated, strict=False)

            # if hparams.val_only or hparams.gt_ft:
            if hparams.val_only:
                loaded_dict = torch.load(hparams.ckpt_path)
                NGP_dict = self.model.state_dict()
                SR_dict = self.SRmodel.state_dict()

                NGP_loaded_dict = {}
                SR_loaded_dict = {}
                for k,v in loaded_dict.items():
                    if k.startswith('model.'):
                        NGP_loaded_dict[k[len('model.'):]] = v
                    if k.startswith('SRmodel.'):
                        SR_loaded_dict[k[len('SRmodel.'):]] = v

                NGP_dict.update(NGP_loaded_dict)
                SR_dict.update(SR_loaded_dict)

                self.model.load_state_dict(NGP_dict)
                if not hparams.TRT_enable:
                    self.SRmodel.load_state_dict(SR_dict)

        else:
            if hparams.val_only:
                loaded_dict = torch.load(hparams.ckpt_path)
                NGP_dict = self.model.state_dict()
                NGP_loaded_dict = {}
                for k,v in loaded_dict.items():
                    if k.startswith('model.'):
                        NGP_loaded_dict[k[len('model.'):]] = v
                NGP_dict.update(NGP_loaded_dict)
                self.model.load_state_dict(NGP_dict)

    def forward(self, batch, split):

        if split=='train' and not self.hparams.super_sampling:  # Pretraining
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        elif split=='train' and self.hparams.super_sampling:    # Joint Training
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        else:                                                   # Eval
            poses = batch['pose']
            if self.hparams.render_low_res:
                directions = self.test_directions
            else:
                directions = self.directions

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        torch.cuda.synchronize()
        start_NGP = timer()

        outs = []
        for i in range(self.hparams.frame_num):
            if split!='train':
                if self.hparams.dataset_name == 'nsvf':
                    # pdb.set_trace()
                    poses = torch.squeeze(poses, dim=0)
                    assert(len(poses.shape)==3)
                elif self.hparams.dataset_name == 'colmap':
                    pass # TODO: to squeeze for multi-frame training in the future

            if self.hparams.training_stage == 'NeRF_pretrain':
                if split == 'train':
                    rays_o, rays_d = get_rays(directions, poses[:,i,:,:])
                    # rays_o, rays_d = get_rays(directions, poses[i])
                else:
                    rays_o, rays_d = get_rays(directions, poses[i])
            else: # SR pretrain or E2E
                rays_o, rays_d = get_rays(directions, poses[i])

            kwargs = {'test_time': split!='train',
                      'random_bg': self.hparams.random_bg}
            if self.hparams.scale > 0.5:
                kwargs['exp_step_factor'] = 1/256
            if self.hparams.use_exposure:
                kwargs['exposure'] = batch['exposure']

            depth_proj = None
            if split=='test':
                start = timer()
                # if i>0:
                if False:
                    depth_prev = outs[-1]['depth'].reshape(1,1,self.train_dataset.low_res_h, self.train_dataset.low_res_w)
                    if False:
                        import matplotlib.pyplot as plt
                        plt.imshow(outs[-1]['rgb'].reshape(200,200,3).detach().cpu().numpy())
                        plt.show()
                        import ipdb;ipdb.set_trace()
                    pose_prev = batch['pose'][:,i-1]
                    pose_curr = batch['pose'][:,i]
                    depth_proj = self.projector(depth_prev, pose_prev, pose_curr)
                    #import ipdb;ipdb.set_trace()

                torch.cuda.synchronize()
                end = timer()
                print(f'NGP frame project {i}: ', (end-start)*1000)

            if split=='test':
                start = timer()

            start = timer()
            out = render(self.model, rays_o, rays_d, depth_proj, **kwargs)
            torch.cuda.synchronize()
            neu_fields_elapsed_time = timer() - start
            out['nerf_t'] = neu_fields_elapsed_time

            # only record time frames accelerated by depth projection
            # if i>0:
            #     self.t.append((end - start))
            # print(f"Shape of out_rgb {out['rgb'].shape}")
            outs.append(out)

            if split=='test':
                torch.cuda.synchronize()
                end = timer()
                print(f'NGP frame render  {i}: ', (end-start)*1000)

        # merge into tensor
        out_merge = {}
        for k in outs[0].keys():
            if k in ['rgb', 'depth', 'opacity', 'feat', ]: #'ts', 'ws', 'rays_a', 'deltas'
                data = torch.cat([o[k] for o in outs])
                data = data.reshape(self.hparams.frame_num, -1, *data.shape[1:])
                out_merge[k]=data
            # TODO: check how to merge rm_samples, vr_samples
            elif k in ['rm_samples', 'vr_samples', 'total_samples', 'ts', 'ws', 'rays_a', 'deltas']:
                out_merge[k] = outs[0][k]
            elif k in ['time', 'nerf_t']:
                out_merge[k] = outs[-1][k]
        out = out_merge

        if split=='test':
            start = timer()

        if split=='train':
            start = timer()

        if self.hparams.super_sampling:
            ### reshape before SR
            # RGB shape: [frame_num, low_h*low_w, 3] -> [1, frame_num, 3, low_h, low_w]
            # Depth shape: [frame_num, low_h*low_w] -> [1, frame_num, 1, low_h, low_w]
            if split == "train":
                out['rgb_low'] = out['rgb'].permute(0, 2, 1).reshape((-1, self.hparams.frame_num, 3, self.train_dataset.patch_res_h, self.train_dataset.patch_res_w))
                if self.hparams.feature_training:
                    out['feat_low'] = out['feat'].permute(0, 2, 1).reshape(
                        (-1, self.hparams.frame_num, 3, self.train_dataset.patch_res_h, self.train_dataset.patch_res_w))


                out['depth'] = out['depth'].reshape((-1, self.hparams.frame_num, 1, self.train_dataset.patch_res_h, self.train_dataset.patch_res_w)).contiguous()
                out['opacity'] = out['opacity'].reshape((-1, self.hparams.frame_num, 1, self.train_dataset.patch_res_h, self.train_dataset.patch_res_w)).contiguous()
                #### DEBUG
                if False:
                    import matplotlib.pyplot as plt
                    fig, axs = plt.subplots(1, 3)
                    axs[0].imshow(out['depth'][0,0,0].detach().cpu().numpy())
                    axs[1].imshow(out['depth'][0,1,0].detach().cpu().numpy())
                    axs[2].imshow(out['depth'][0,2,0].detach().cpu().numpy())
                    plt.show()
                    fig, axs = plt.subplots(1, 3)
                    axs[0].imshow(out['rgb_low'][0,0].permute(1,2,0).detach().cpu().numpy())
                    axs[1].imshow(out['rgb_low'][0,1].permute(1,2,0).detach().cpu().numpy())
                    axs[2].imshow(out['rgb_low'][0,2].permute(1,2,0).detach().cpu().numpy())
                    plt.show()
                    fig, axs = plt.subplots(1, 3)
                    axs[0].imshow(out['feat_low'][0,0].permute(1,2,0).detach().cpu().numpy())
                    axs[1].imshow(out['feat_low'][0,1].permute(1,2,0).detach().cpu().numpy())
                    axs[2].imshow(out['feat_low'][0,2].permute(1,2,0).detach().cpu().numpy())
                    plt.show()
            else:
                # pdb.set_trace()
                out['rgb_low'] = out['rgb'].permute(0, 2, 1).reshape(
                    (-1, self.hparams.frame_num, 3, self.test_dataset.low_res_h, self.test_dataset.low_res_w))
                if self.hparams.feature_training:
                    out['feat_low'] = out['feat'].permute(0, 2, 1).reshape(
                        (-1, self.hparams.frame_num, 3, self.test_dataset.low_res_h, self.test_dataset.low_res_w))
                    # out['feat_low'][:, 1:3] = 0

                out['depth'] = out['depth'].reshape(
                    (-1, self.hparams.frame_num, 1, self.test_dataset.low_res_h, self.test_dataset.low_res_w)).contiguous()
                out['opacity'] = out['opacity'].reshape(
                    (-1, self.hparams.frame_num, 1, self.test_dataset.low_res_h, self.test_dataset.low_res_w)).contiguous()


            # TODO: only support batch_size=1 for now
            assert(out['rgb_low'].shape[0]==1)
            ### SR
            if self.hparams.feature_training:
                out['rgb'], out['net_t'] = self.SRmodel(out['rgb_low'], out['depth'], batch['pose'],
                        batch['offset'], K=batch['low_res_K'], feat=out['feat_low']) #mask = out['opacity'][:,0]
            else: # multiple_frames
                out['rgb'] = self.SRmodel(out['rgb_low'], out['depth'], batch['pose'],
                                          batch['offset'], K=batch['low_res_K'])

            #### DEBUG
            if False:
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(1, 1)
                axs.imshow(out['rgb'][0].permute(1,2,0).detach().float().cpu().numpy())
                plt.show()
                import ipdb;ipdb.set_trace()

            ### reshape after SR
            # RGB shape: [1, 3, high_h, high_w] -> [high_h*high_w, 3]
            # No need to output depth
            # Depth shape: [1, 1, low_h, low_w] -> [low_h*low_w] # cause no high res depth supervision
            if split == "train":
                out['rgb_vis'] = out['rgb']
                out['rgb'] = out['rgb'].reshape(
                    (3, self.train_dataset.patch_high_res_h * self.train_dataset.patch_high_res_w)).permute(1, 0).contiguous()
                # [Bs==1, frame_num, C==3, h, w] --> [frame_num, h*w, C==3]
                out['rgb_low'] = out['rgb_low'][:, 0]
                out['rgb_low'] = out['rgb_low'].reshape(
                    (1, 3, self.train_dataset.patch_res_h * self.train_dataset.patch_res_w)).permute(0, 2, 1).contiguous()

            else:
                # out['rgb_vis'] = out['rgb']
                out['rgb'] = out['rgb'].reshape(
                    (3, self.test_dataset.high_res_h * self.test_dataset.high_res_w)).permute(1, 0).contiguous()

        return out

    def setup(self, stage): # set up train&val&test dataset
        dataset = dataset_dict[self.hparams.dataset_name]

        ### Trainset Settings
        kwargs = {'root_dir': self.hparams.root_dir}

        kwargs['downsample'] = self.hparams.downsample
        kwargs['super_sampling_factor'] = self.hparams.super_sampling_factor if self.hparams.super_sampling else None
        kwargs['frame_num'] = self.hparams.frame_num
        kwargs['training_stage'] = self.hparams.training_stage
        kwargs['patch_size'] = self.hparams.patch_size
        kwargs['val_only'] = self.hparams.val_only

        print(f"super_sampling_factor: {kwargs['super_sampling_factor']}")

        self.train_dataset = dataset(split=self.hparams.split if not self.hparams.render_traj else 'test_traj', **kwargs)

        if not self.hparams.super_sampling:
            self.train_dataset.batch_size = self.hparams.batch_size
            self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy


        ### Testset Settings
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.render_downsample} # change to 0.25 if render low-res. img of train set
        kwargs['super_sampling_factor'] = self.hparams.super_sampling_factor if self.hparams.super_sampling else None
        kwargs['frame_num'] = self.hparams.frame_num
        kwargs['patch_size'] = self.hparams.patch_size

        if self.hparams.render_traj:
            self.test_dataset = dataset(split='test_traj', **kwargs)
        else:
            self.test_dataset = dataset(split='test', **kwargs) # 'testtrain' is to render low-res. img as trainset for SR


    def _configure_optimizers_NeRF_pretrain(self):
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('test_directions', self.test_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))

        load_ckpt(self.model, self.hparams.weight_path)

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT'] and 'SR' not in n:
                net_params += [p]

        opts = []
        # NGP param. optimizer
        self.opts = FusedAdam(net_params, self.hparams.lr, eps=1e-15)
        opts += [self.opts]

        if self.hparams.optimize_ext:
            opts += [FusedAdam([self.dR, self.dT], 1e-6)] # learning rate is hard-coded
        net_sch = CosineAnnealingLR(self.opts,
                                    self.hparams.num_epochs,
                                    self.hparams.lr/30.)

        return opts, [net_sch]

    def _configure_optimizers_E2E_joint_training(self):
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('test_directions', self.test_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        load_ckpt(self.model, self.hparams.weight_path)

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT'] and 'SR' not in n:
                # p.requires_grad = False
                net_params += [p]

        self.opts = FusedAdam([{'params':net_params, 'lr':self.hparams.lr, 'eps':1e-15},
                               {'params':self.SRmodel.parameters(), 'lr':self.hparams.lr_SR}])

        # scheduler1 = CosineAnnealingLR(self.opts, T_max=100, eta_min=5e-5)
        # scheduler2 = MultiStepLR(self.opts, milestones=[5, 10], gamma=0.5)
        # net_sch = ChainedScheduler([scheduler1, scheduler2])

        scheduler1 = CosineAnnealingLR(self.opts, T_max=100, eta_min=5e-5)
        scheduler2 = ConstantLR(self.opts, factor=1, total_iters=100)
        net_sch = SequentialLR(self.opts, schedulers=[scheduler1, scheduler2], milestones=[300])


        return ([self.opts], [net_sch])

    def configure_optimizers(self):
        if self.training_stage == 'NeRF_pretrain':
            return self._configure_optimizers_NeRF_pretrain()
        elif self.training_stage == 'End2End':
            return self._configure_optimizers_E2E_joint_training()
        else:
            raise NotImplementedError()

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=8,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=1,
                          batch_size=None,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=1,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        print("wh:",self.test_dataset.img_wh)
        self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
                                        self.poses,
                                        self.train_dataset.img_wh)

        ### logger: to specify dataset_name and scene_name
        tensorboard = self.logger.experiment
        tensorboard.add_text("dataset", hparams.dataset)
        tensorboard.add_text("scene", hparams.exp_name)

        if hparams.distortion_loss_w > 0:
            tensorboard.add_text("d_loss", hparams.dataset)

        if hparams.frame_num > 1 and hparams.super_sampling:
            self.projector = Projection(K=self.train_dataset.K, h=self.test_dataset.low_res_h, w=self.test_dataset.low_res_w)


    def on_train_epoch_start(self):
        if not self.hparams.no_save_test:
            self.save_base_dir = os.path.join(self.logger.root_dir, f"version_{self.logger.version}", 'results')
            self.val_dir = self.save_base_dir
            os.makedirs(self.val_dir, exist_ok=True)

        if self.current_epoch%10 == 0:
            ckpt_name = os.path.join(self.val_dir, f"latest_ckpt.ckpt")
            self.trainer.save_checkpoint(ckpt_name)
            print("saving pytorch_lightning ckpt...")

    def training_step(self, batch, batch_nb, *args):
        if self.global_step%self.update_interval == 0:
            self.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
                                           warmup=self.global_step<self.warmup_steps,
                                           erode=self.hparams.dataset_name=='colmap')

        # torch.cuda.synchronize()
        # start = timer()
        # print(batch['rgb'].shape)
        results = self(batch, split='train')
        # torch.cuda.synchronize()
        # forward_time = timer() - start
        #
        # # print(f"Whole pipeline forward time: {forward_time*1000} ms")
        DEBUG =False
        if DEBUG and batch_nb==0:
            # rgb_gt = rearrange(batch['rgb'] , '(h w) c -> h w c', h=600)
            # rgb_gt = torch.clamp(rgb_gt, 0., 1.)
            # rgb_gt = rgb_gt.cpu().numpy()
            # rgb_gt = (rgb_gt * 255).astype(np.uint8)
            # os.makedirs(os.path.join(self.val_dir, 'debug'), exist_ok=True)
            # imageio.imsave(os.path.join(self.val_dir, 'debug', f'debug_{batch_nb:03d}_rgb_gt.png'), rgb_gt)

            rgb_pred = rearrange(results['rgb'], '(h w) c -> h w c', h=600)
            rgb_pred = rgb_pred.detach()
            rgb_pred = torch.clamp(rgb_pred, 0., 1.)
            rgb_pred = rgb_pred.cpu().numpy()
            rgb_pred = (rgb_pred * 255).astype(np.uint8)
            os.makedirs(os.path.join(self.val_dir, 'debug'), exist_ok=True)
            imageio.imsave(os.path.join(self.val_dir, 'debug', f'debug_{batch_nb:03d}_rgb_pred.png'), rgb_pred)

        loss_d = self.loss(results, batch)
        if loss_d['rgb'].sum().item() == 0:
            pdb.set_trace()
        if self.hparams.use_exposure:
            zero_radiance = torch.zeros(1, 3, device=self.device)
            unit_exposure_rgb = self.model.log_radiance_to_rgb(zero_radiance,
                                    **{'exposure': torch.ones(1, 1, device=self.device)})
            loss_d['unit_exposure'] = \
                0.5*(unit_exposure_rgb-self.train_dataset.unit_exposure_rgb)**2
        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
            if False:
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(1, 3)
                axs[0].imshow(results['rgb'].clamp(min=0.,max=1.).reshape(600,600,3).detach().float().cpu().numpy())
                axs[1].imshow(batch['rgb'].reshape(600,600,3).detach().float().cpu().numpy())
                axs[2].imshow((batch['rgb']-results['rgb']).reshape(600,600,3).detach().float().cpu().numpy())
                plt.show()
                import ipdb;ipdb.set_trace()
        if self.hparams != 'SR_pretrain':
            self.log('lr/NGP_lr', self.opts.param_groups[0]['lr'])
        if self.hparams.super_sampling:
            if self.hparams != 'SR_pretrain':
                self.log('lr/SR_lr', self.opts.param_groups[1]['lr'])
            else:
                self.log('lr/SR_lr', self.opts.param_groups[0]['lr'])
        self.log('train/loss', loss)
        # ray marching samples per ray (occupied space on the ray)
        self.log('train/rm_s', results['rm_samples']/len(batch['rgb']), True)
        # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
        self.log('train/vr_s', results['vr_samples']/len(batch['rgb']), True)
        self.log('train/psnr', self.train_psnr, True)

        return loss

    def on_validation_start(self):
        if hparams.super_sampling:
            print("wh:", self.test_dataset.img_wh)
            self.SRmodel.im_width = self.test_dataset.img_wh[0]
            self.SRmodel.im_height = self.test_dataset.img_wh[1]

        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('test_directions', self.test_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        if hparams.frame_num > 1 and hparams.super_sampling:
            self.projector = Projection(K=self.train_dataset.K, h=self.test_dataset.low_res_h, w=self.test_dataset.low_res_w)

        torch.cuda.empty_cache()
        if not self.hparams.no_save_test:
            self.save_base_dir = os.path.join(self.logger.root_dir, f"version_{self.logger.version}", 'results')
            self.val_dir = self.save_base_dir
            os.makedirs(self.val_dir, exist_ok=True)
        else:
            if self.hparams.render_traj:
                self.save_base_dir = os.path.join(self.logger.root_dir, f"version_{self.logger.version}", 'eval_traj')
            else:
                self.save_base_dir = os.path.join(self.logger.root_dir, f"version_{self.logger.version}", 'eval')
            self.val_dir = self.save_base_dir
            os.makedirs(self.val_dir, exist_ok=True)

        result_filename = os.path.join(self.logger.root_dir, f"version_{self.logger.version}", 'results.txt')
        with open(result_filename, 'w') as f:
            f.write(f"feature training: {self.hparams.feature_training} \n")
            f.write(f"frame num: {self.hparams.frame_num} \n")

        self.elapsed_time_ms = 0
        self.t =[]

    def validation_step(self, batch, batch_nb):
        if not self.hparams.render_traj:
            rgb_gt = batch['rgb']
            torch.cuda.synchronize()

        speed_test = True
        warm_up_start = timer()
        if speed_test and batch_nb==0:
            for _ in range(10):
                results = self(batch, split='test')
                torch.cuda.synchronize()
            warm_up_end = timer()
            print(f"Warm up time:{(warm_up_end-warm_up_start)}s")

        start = timer()
        results = self(batch, split='test')
        torch.cuda.synchronize()
        end = timer()

        logs = {}
        logs['total_samples'] = results['total_samples'].cpu().type(torch.FloatTensor)
        logs['time'] = results['time']
        logs['net_t'] = results.get('net_t', 0)
        logs['nerf_t'] = results['nerf_t']

        if self.hparams.render_traj and self.hparams.save_traj_img:
            # pass
            w, h = self.test_dataset.img_wh
            rgb_pred = rearrange(results['rgb'], '(h w) c -> h w c', h=h)
            rgb_pred = torch.clamp(rgb_pred, 0., 1.)
            rgb_pred = rgb_pred.cpu().numpy()
            rgb_pred = (rgb_pred * 255).astype(np.uint8)
            imageio.imsave(os.path.join(self.val_dir, f'{batch_nb:03d}_rgb.png'), rgb_pred)

        fig_save = False
        if fig_save:
            idx = batch['img_idxs']
            low_depth_npy = results['depth'].cpu().numpy()
            low_feat_npy = results['feat_low'].cpu().numpy()

            np.save(os.path.join(self.val_dir, f'{idx:03d}_depth_low.npy'), low_depth_npy[0,0])
            np.save(os.path.join(self.val_dir, f'{idx:03d}_feat_low.npy'), low_feat_npy[0,0])

        if not self.hparams.render_traj:
            self.elapsed_time_ms += (end - start) * 1000

            # compute each metric per image
            rgb_pred = results['rgb'].clamp(0.,1.)
            gt_mask = rearrange(batch['mask'], 'a -> a 1')
            rgb_pred = rgb_pred * gt_mask + (1 - gt_mask)

            if self.hparams.training_stage == 'NeRF_pretrain':
                self.val_psnr(results['rgb'].clamp(0.,1.), rgb_gt)
            else:
                self.val_psnr(rgb_pred, rgb_gt)
            logs['psnr'] = self.val_psnr.compute()
            self.val_psnr.reset()

            w, h = self.test_dataset.img_wh

            if self.hparams.training_stage == 'NeRF_pretrain':
                rgb_pred = rearrange(results['rgb'], '1 (h w) c -> 1 c h w', h=h)
            else:
                rgb_pred = rearrange(rgb_pred, '(h w) c -> 1 c h w', h=h).to(rgb_gt.dtype)

            rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)

            self.val_ssim(rgb_pred, rgb_gt)
            logs['ssim'] = self.val_ssim.compute()
            self.val_ssim.reset()
            #
            if self.hparams.eval_lpips:
                self.val_lpips(torch.clip(rgb_pred*2-1, -1, 1),
                               torch.clip(rgb_gt*2-1, -1, 1))
                logs['lpips'] = self.val_lpips.compute()
                self.val_lpips.reset()

            if not self.hparams.no_save_test: # save test image to disk
                idx = batch['img_idxs']
                if self.hparams.training_stage == 'NeRF_pretrain':
                    rgb_pred = rearrange(results['rgb'], '1 (h w) c -> h w c', h=h) # for saving only
                else:
                    rgb_pred = rearrange(rgb_pred, '1 c h w -> h w c', h=h)

                rgb_pred = rgb_pred.cpu().numpy()
                rgb_pred = (rgb_pred*255).astype(np.uint8)

                rgb_gt = rearrange(rgb_gt, '1 c h w -> h w c', h=h)
                rgb_gt = rgb_gt.cpu().numpy()
                rgb_gt = (rgb_gt*255).astype(np.uint8)

                diff_rgb_mask = np.isclose(rgb_pred, rgb_gt)

                # # visualize the first image only
                # rgb_low = rearrange(results['rgb_low'][:,0], '1 c h w -> h w c', h=self.test_dataset.low_res_h)
                # pdb.set_trace()
                if self.hparams.super_sampling:
                    rgb_low = rearrange(results['rgb_low'][:, 0, :, :, :], '1 c h w -> h w c', h=self.test_dataset.low_res_h)
                    rgb_low = torch.clamp(rgb_low, 0., 1.)
                    rgb_low = rgb_low.cpu().numpy()
                    rgb_low = (rgb_low*255).astype(np.uint8)
                    imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_rgb_low.png'), rgb_low)
                    # pdb.set_trace()

                    # visualize the first image only
                    if self.hparams.feature_training:
                        feat_low = rearrange(results['feat_low'][:,0], '1 c h w -> h w c', h=self.test_dataset.low_res_h)
                        feat_low = torch.clamp(feat_low, 0., 1.)
                        feat_low = feat_low.cpu().numpy()
                        feat_low = (feat_low*255).astype(np.uint8)
                        imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_feat_low.png'), feat_low)

                # depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
                # depth = depthScaling(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
                if not self.hparams.super_sampling:
                    depth = rearrange(results['depth'].cpu().numpy(), '1 (h w) -> h w', h=h)
                    opacity = rearrange(results['opacity'].cpu().numpy(), '1 (h w) -> h w', h=h)
                    opacity = (opacity * 255).astype(np.uint8)

                multi_frame = False
                if not multi_frame:
                    if not self.hparams.super_sampling:
                        imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_rgb.png'), rgb_pred)
                        np.save(os.path.join(self.val_dir, f'{idx:03d}_d.npy'), depth)
                        imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_mask.png'), opacity)
                    else:
                        imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_rgb.png'), rgb_pred)
                else:
                    prefix = 0
                    seq_length = 10
                    file_idx = idx // seq_length
                    frame_idx = idx % seq_length

                    np.save(os.path.join(self.val_dir, f'0_{file_idx:04d}-{frame_idx:02d}_d.npy'), depth)

                    imageio.imsave(os.path.join(self.val_dir, f'0_{file_idx:04d}-{frame_idx:02d}_rgb.png'), rgb_pred)
                    # imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_d.png'), depth)
                    imageio.imsave(os.path.join(self.val_dir, f'0_{file_idx:04d}-{frame_idx:02d}_mask.png'), opacity)

            return logs

        else:
            return logs

    def validation_epoch_end(self, outputs):
        # print(sum(self.t)/len(self.t))
        # pdb.set_trace()
        # print(f"num of val frame:{len(outputs)}")
        # print(f"{self.elapsed_time_ms/(len(outputs))}ms/frame")
        # pdb.set_trace()

        nerf_t = [x['nerf_t'] for x in outputs]
        net_t = [x['net_t'] for x in outputs]
        print('neural_fields_mean_t:', sum(nerf_t) / len(nerf_t) * 1e3, 'ms')
        print('neural_renderer_mean_t:', sum(net_t) / len(net_t) * 1e3, 'ms')
        print('FPS:', 1/(sum(nerf_t) / len(nerf_t) + sum(net_t) / len(net_t)) )

        # logging
        if not self.hparams.render_traj:
            psnrs = torch.stack([x['psnr'] for x in outputs])
            mean_psnr = all_gather_ddp_if_available(psnrs).mean()
            self.log('test/psnr', mean_psnr, True)

            ssims = torch.stack([x['ssim'] for x in outputs])
            mean_ssim = all_gather_ddp_if_available(ssims).mean()
            self.log('test/ssim', mean_ssim)

            if self.hparams.eval_lpips:
                lpipss = torch.stack([x['lpips'] for x in outputs])
                mean_lpips = all_gather_ddp_if_available(lpipss).mean()
                self.log('test/lpips_vgg', mean_lpips)
            # samples = torch.stack([x['total_samples'] for x in outputs])
            # print(f'Average total samples:{samples.mean().item()}')
            #
            # time =[x['time'] for x in outputs]
            # print(f'Average time to render one frame:{(sum(time)/len(time))* 1000} ms')
            # mean_psnr = all_gather_ddp_if_available(psnrs).mean()
            # self.log('test/psnr', mean_psnr, True)

            result_filename = os.path.join(self.logger.root_dir, f"version_{self.logger.version}", 'results.txt')
            with open(result_filename, 'a+') as f:
                f.write(f"{mean_psnr} {mean_ssim} {mean_lpips} \n")


    def on_test_start(self) -> None:
        self.elapsed_time_ms = 0

    def test_step(self, batch, batch_nb):
        start = timer()
        results = self(batch, split='test')
        torch.cuda.synchronize()
        end = timer()

        self.elapsed_time_ms += (end - start) * 1000

        if not self.hparams.no_save_test: # save test image to disk
            idx = batch['img_idxs']
            w, h = self.test_dataset.img_wh # change from 'train_dataset' to 'test_dataset'
            rgb_pred = rearrange(results['rgb'], '(h w) c -> h w c', h=h)
            if self.hparams.super_sampling:
                rgb_pred = torch.clamp(rgb_pred, 0., 1.)
            rgb_pred = rgb_pred.cpu().numpy()
            rgb_pred = (rgb_pred*255).astype(np.uint8)
            imageio.imsave(os.path.join(f'./debug/{idx:03d}_rgb.png'), rgb_pred)

        return []

    def test_epoch_end(self, outputs) -> None:
        # benchmark
        print(f"num of val frame:{len(outputs)}")
        print(f"{self.elapsed_time_ms/(len(outputs))}ms/frame")

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items



if __name__ == '__main__':
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')

    print(f"Scene {hparams.exp_name}")
    if hparams.val_only:
        # dirpath = f'ckpts/{hparams.dataset_name}/{hparams.exp_name}' if hparams.log2_T == 19 else f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/{hparams.log2_T}'
        # ckpt_cb = ModelCheckpoint(dirpath=dirpath,
        #                           filename='{epoch:d}',
        #                           save_weights_only=True,
        #                           every_n_epochs=hparams.num_epochs,
        #                           save_on_train_epoch_end=True,
        #                           save_top_k=-1)
        # callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]
        #
        # logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}",
        #                            name=hparams.exp_name,
        #                            default_hp_metric=False)
        #
        # trainer = Trainer(max_epochs=hparams.num_epochs,
        #                   check_val_every_n_epoch=5,  # if hparams.feature_training else hparams.num_epochs,
        #                   callbacks=callbacks,
        #                   logger=logger,
        #                   enable_model_summary=False,
        #                   accelerator='gpu',
        #                   devices=hparams.num_gpus,
        #                   strategy=DDPPlugin(find_unused_parameters=False)
        #                   if hparams.num_gpus > 1 else None,
        #                   num_sanity_val_steps=2,  # if hparams.val_only else 0,
        #                   precision=16)

        system = NeRFSystem(hparams)
        # TODO: three stage validation
        # need to register "directions", "test_directions" and "poses"
        # system.register_buffer('directions', torch.ones(40000,3))
        # system.register_buffer('test_directions', torch.ones(40000,3))
        # system.register_buffer('poses', torch.ones(1100,hparams.frame_num, 3, 4))

        dirpath = f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/End2End_direct' if hparams.feature_training else \
            f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/no_feat/End2End_direct'

        print(dirpath)
        ckpt_cb = ModelCheckpoint(monitor='test/psnr',
                                  mode='max',
                                  dirpath=dirpath,
                                  filename='{epoch:d}',
                                  save_weights_only=True,
                                  every_n_epochs=20,
                                  save_on_train_epoch_end=True,
                                  save_top_k=1,
                                  save_last=True)

        callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]
        logger = TensorBoardLogger(
            save_dir=f"logs/{hparams.dataset_name}/{hparams.exp_name}" if hparams.feature_training else f"logs/{hparams.dataset_name}/{hparams.exp_name}/no_feat",
            name="End2End_direct",
            default_hp_metric=False)

        trainer = Trainer(max_epochs=hparams.num_epochs,
                          check_val_every_n_epoch=50 if "Tanks" in hparams.root_dir else 20,
                          # if hparams.feature_training else hparams.num_epochs,
                          callbacks=callbacks,
                          logger=logger,
                          enable_model_summary=False,
                          accelerator='gpu',
                          devices=hparams.num_gpus,
                          strategy=DDPPlugin(find_unused_parameters=False)
                          if hparams.num_gpus > 1 else None,
                          num_sanity_val_steps=2,  # if hparams.val_only else 0,
                          precision=16,)

        trainer.validate(system)
        # video synthesis
        # os.makedirs('output', exist_ok=True)
        # os.system(f"ffmpeg -framerate 30  -i ./{system.val_dir}/%03d_rgb.png -q 2  ./output/{hparams.exp_name}.mp4")

    else:
        if hparams.complete_pipeline:
            if hparams.direct_E2E:
                print("Training mode: Direct end to end training")
                ###End to end training directly
                hparams.training_stage = 'End2End'
                system = NeRFSystem(hparams)

                dirpath = f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/End2End_direct'

                if hparams.distortion_loss_w > 0:
                    dirpath+='_dloss'

                print(dirpath)
                ckpt_cb = ModelCheckpoint(monitor='test/psnr',
                                          mode='max',
                                          dirpath=dirpath,
                                          filename='{epoch:d}',
                                          save_weights_only=False,
                                          every_n_epochs=20,
                                          save_on_train_epoch_end=True,
                                          save_top_k=1,
                                          save_last=True)

                callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

                logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}/{hparams.exp_name}" ,
                                           name="End2End_direct",
                                           default_hp_metric=False)

                trainer = Trainer(max_epochs=hparams.num_epochs,
                                  check_val_every_n_epoch=50 if "Tanks" in hparams.root_dir else 1,  # if hparams.feature_training else hparams.num_epochs,
                                  callbacks=callbacks,
                                  logger=logger,
                                  enable_model_summary=False,
                                  accelerator='gpu',
                                  devices=hparams.num_gpus,
                                  strategy=DDPPlugin(find_unused_parameters=False)
                                  if hparams.num_gpus > 1 else None,
                                  num_sanity_val_steps=2,
                                  precision=16)

                trainer = Trainer(max_epochs=hparams.num_epochs,
                                  check_val_every_n_epoch=20,
                                  callbacks=callbacks,
                                  logger=logger,
                                  enable_model_summary=False,
                                  accelerator='gpu',
                                  devices=hparams.num_gpus,
                                  strategy=DDPPlugin(find_unused_parameters=False)
                                  if hparams.num_gpus > 1 else None,
                                  num_sanity_val_steps=2,  # if hparams.val_only else 0,
                                  precision=16)

                trainer.fit(system)

        else:
            if hparams.training_stage=='NeRF_pretrain':
                system = NeRFSystem(hparams)
                dirpath = f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/NeRF_Pretrain'
                ckpt_cb = ModelCheckpoint(dirpath=dirpath,
                                          filename='{epoch}',
                                          save_weights_only=True,
                                          every_n_epochs=hparams.num_epochs,
                                          save_on_train_epoch_end=False,
                                          save_last=True)
                callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

                logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}/{hparams.exp_name}",
                                           name="NeRF_Pretrain",
                                           default_hp_metric=False)

                trainer = Trainer(max_epochs=hparams.num_epochs,
                                  check_val_every_n_epoch=hparams.num_epochs,
                                  callbacks=callbacks,
                                  logger=logger,
                                  enable_model_summary=False,
                                  accelerator='gpu',
                                  devices=hparams.num_gpus,
                                  strategy=DDPPlugin(find_unused_parameters=False)
                                  if hparams.num_gpus > 1 else None,
                                  num_sanity_val_steps=2,
                                  precision=16)

                trainer.fit(system, ckpt_path=hparams.ckpt_path)


    if not hparams.val_only: # save slimmed ckpt for the last epoch
        # if hparams.training_stage == 'NeRF_pretrain':
        ckpt_ = slim_ckpt(f'{dirpath}/last.ckpt', save_poses=hparams.optimize_ext)
        torch.save(ckpt_, f'{dirpath}/epoch={hparams.num_epochs-1}_slim_feat.ckpt')
        # else:
        #     ckpt_ = slim_ckpt(f'{dirpath}/last.ckpt', save_poses=hparams.optimize_ext)
        #     torch.save(ckpt_, f'{dirpath}/epoch={hparams.num_epochs-1}_slim.ckpt')

