import torch
from torch import nn
from timeit import default_timer as timer

class Projection(nn.Module):
    def __init__(self, K=None, h=None, w=None):
        super().__init__()
        if K is not None:
            self.K = K.view(-1,3,3).to(torch.float32)
            Ki = torch.inverse(K)

            #self.im_height = torch.tensor([h]).long()
            #self.im_width = torch.tensor([w]).long() 
            im_height = h
            im_width = w
            # self.im_height = K[0,2].long()*2 
            # self.im_width = K[0,2].long()*2 
            # im_height = self.im_height.item()
            # im_width = self.im_width.item()

            # low-res ray
            i, j = torch.meshgrid(torch.linspace(0, im_width-1, im_width), torch.linspace(0, im_height-1, im_height)) # pytorch's meshgrid has indexing='ij'
            i = i.t().to(K.device)
            j = j.t().to(K.device)
            self.i = i
            self.j = j
            K = K.view(3,3)
            ray = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
            self.ray = ray.reshape(1,-1,3).to(torch.float32)

            self.K = self.K.cuda()
            self.ray = self.ray.cuda()
            self.last_row = torch.tensor([0.,0.,0.,1]).reshape(1,1,4).float().cuda()

        self.ones = torch.ones(1,h*w, 1).float().cuda()
        self.depth_proj = torch.zeros(1,1,h,w).float().cuda()

    def get_K(self, im_height, im_width, offset, high_res=False):
        B = offset.shape[0]
        if high_res:
            K = self.K_highres.view(3,3)
            offset_ = offset * 4
        else:
            K = self.K.view(3,3)
            offset_ = offset
        K = K.to(offset.device).unsqueeze(0).repeat(B, 1,1)
        # K[:,0,2] = im_width / 2 - offset[:,0] * K[:,0,2] 
        # K[:,1,2] = im_height / 2 - offset[:,1] * K[:,1,2]
        K[:,0,2] -= offset_[:,1]  
        K[:,1,2] -= offset_[:,0] 
        return K

    # def get_ray(self, im_height, im_width, offset, high_res=False):
    #     K = self.get_K(im_height, im_width, offset, high_res)
    #     B = offset.shape[0]
    #     i, j = torch.meshgrid(torch.linspace(0, im_width-1, im_width), torch.linspace(0, im_height-1, im_height))  # pytorch's meshgrid has indexing='ij'
    #     i = i.t()
    #     j = j.t()
    #     i = i.to(offset.device).unsqueeze(0).repeat(B,1,1)
    #     j = j.to(offset.device).unsqueeze(0).repeat(B,1,1)
    #     ray = torch.stack([(i-K[:,0,2].reshape(B,1,1))/K[:,0,0].reshape(B,1,1), (j-K[:,1,2].reshape(B,1,1))/K[:,1,1].reshape(B,1,1), torch.ones_like(i)], -1)
    #     return ray.reshape(B,-1,3).to(torch.float32)

    def unproject(self, depth, pose, offset=None, high_res=False):
        #ray = self.get_ray(depth.shape[-2], depth.shape[-1], offset, high_res)
        ray = self.ray
        bs = depth.shape[0]

        xyz = depth.reshape(bs,-1,1) * ray.to(depth.device)

        # c2w
        if pose.shape[1]==3:
            pose = torch.cat((pose, torch.tensor([0.,0.,0.,1]).reshape(1,1,4).to(pose.dtype).to(pose.device)), 1)
        xyz = torch.cat((xyz, torch.ones_like(xyz[...,-1:])), -1)
        xyz = (pose @ xyz.transpose(1,2)).transpose(1,2)
        xyz = xyz[...,0:3]

        return xyz

    def project(self, xyz, pose, im_height, im_width, offset, high_res=False):
        # if not high_res:
        #     K = self.K.to(xyz.device)
        # else:
        #     K = self.K_highres.to(xyz.device)
        if pose.shape[1]==3:
            pose = torch.cat((pose, torch.tensor([0.,0.,0.,1]).reshape(1,1,4).to(pose.dtype).to(pose.device)), 1)
        #K = self.get_K(im_height, im_width, offset, high_res)
        K = self.K
        bs = xyz.shape[0]
  
        # w2c
        xyz = torch.cat((xyz, torch.ones_like(xyz[...,-1:])), -1)
        xyz = (torch.inverse(pose) @ xyz.transpose(1,2)).transpose(1,2)
        xyz = xyz[...,0:3]

        Kt = K.transpose(1,2)
        uv = torch.bmm(xyz, Kt.to(xyz.device))
  
        # import ipdb;ipdb.set_trace()
        d = uv[:,:,2:3]
  
        # avoid division by zero
        uv = uv[:,:,:2] / (torch.nn.functional.relu(d) + 1e-12)
        return uv, d

    def unproject_project(self, depth, pose, offset=None, high_res=False):
        #ray = self.get_ray(depth.shape[-2], depth.shape[-1], offset, high_res)
        ray = self.ray
        bs = depth.shape[0]

        xyz = depth.reshape(bs,-1,1) * ray

        # c2w
        #if pose.shape[1]==3:
        #    pose = torch.cat((pose, torch.tensor([0.,0.,0.,1]).reshape(1,1,4).to(pose.dtype).to(pose.device)), 1)

        #xyz = torch.cat((xyz, torch.ones_like(xyz[...,-1:])), -1)
        xyz = torch.cat((xyz, self.ones), -1)
        xyz = (pose @ xyz.transpose(1,2)).transpose(1,2)
        xyz = xyz[...,0:3]

        K = self.K
        Kt = K.transpose(1,2)
        uv = torch.bmm(xyz, Kt.to(xyz.device))
  
        # import ipdb;ipdb.set_trace()
        d = uv[:,:,2:3]
  
        # avoid division by zero
        uv = uv[:,:,:2] / (torch.nn.functional.relu(d) + 1e-12)
        return uv, d

    def project(self, xyz, pose, im_height, im_width, offset, high_res=False):
        # if not high_res:
        #     K = self.K.to(xyz.device)
        # else:
        #     K = self.K_highres.to(xyz.device)
        if pose.shape[1]==3:
            pose = torch.cat((pose, torch.tensor([0.,0.,0.,1]).reshape(1,1,4).to(pose.dtype).to(pose.device)), 1)
        #K = self.get_K(im_height, im_width, offset, high_res)
        K = self.K
        bs = xyz.shape[0]
  
        # w2c
        xyz = torch.cat((xyz, torch.ones_like(xyz[...,-1:])), -1)
        xyz = (torch.inverse(pose) @ xyz.transpose(1,2)).transpose(1,2)
        xyz = xyz[...,0:3]

        Kt = K.transpose(1,2)
        uv = torch.bmm(xyz, Kt.to(xyz.device))
  
        # import ipdb;ipdb.set_trace()
        d = uv[:,:,2:3]
  
        # avoid division by zero
        uv = uv[:,:,:2] / (torch.nn.functional.relu(d) + 1e-12)
        return uv, d
    # project a previous frame to current frame
    # depth:     1x1xHxW
    # pose:      1x3x4 
    # pose_next: 1x3x4
    # offset:    1x2
    def forward(self, depth, pose, pose_next, offset=torch.zeros(1,2)):
        start = timer()
        _,_,H,W = depth.shape
        #xyz = self.unproject(depth, pose, offset=offset, high_res=False)
        if pose.shape[1]==3:
            pose = torch.cat((pose, torch.tensor([0.,0.,0.,1]).reshape(1,1,4).to(pose.dtype).to(pose.device)), 1)
        if pose_next.shape[1]==3:
            pose_next = torch.cat((pose_next, torch.tensor([0.,0.,0.,1]).reshape(1,1,4).to(pose.dtype).to(pose.device)), 1)
        pose_to_next = torch.inverse(pose_next) @ pose
        uv0,d0 = self.unproject_project(depth, pose_to_next, offset=offset, high_res=False)
        ## DEBUG
        if False:
            d0_upsampled = self.depth_zero_upsampling(depth[:,0]) 
            xyz2 = self.unproject(d0_upsampled, pose[:,0], offset=offset, high_res=False)
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
            import ipdb;ipdb.set_trace()
        #torch.cuda.synchronize()
        end = timer()
        print(f'+++++++ projector project, unproject ', (end-start)*1000)
        ###
        # project to frame 0
        #uv0, d0 = self.project(xyz, pose_next, H, W, offset=offset, high_res=False)
        uv0 = torch.round(uv0).to(torch.long)
        #u_mask = torch.logical_and(uv0[...,0]>=0 , uv0[...,0]<H)
        #v_mask = torch.logical_and(uv0[...,1]>=0 , uv0[...,1]<W)
        #uv_mask = torch.logical_and(u_mask, v_mask)
        start = timer()
        uv_mask = torch.logical_and(torch.logical_and(uv0[...,0]>=0 , uv0[...,0]<W),
                                   torch.logical_and(uv0[...,1]>=0 , uv0[...,1]<H))

        #for k in range(depth.shape[0]):
        uv0_masked = uv0[0,uv_mask[0]]
        uv_round = uv0_masked
        #depth_proj = torch.zeros_like(depth) # B,C,H,W
        depth_proj = self.depth_proj.zero_()
        depth = depth.reshape(*depth.shape[0:2],-1) # B,C,H,W -> B,C,H*W
        depth = depth.permute(0,2,1)
        depth_proj[0,:,uv_round[...,1],uv_round[...,0]] = depth[uv_mask[0:1]].T
        if False:
            import matplotlib.pyplot as plt
            plt.imshow(depth_proj[0,0].detach().cpu().numpy())
            plt.show()
            import ipdb;ipdb.set_trace()

        #torch.cuda.synchronize()
        end = timer()
        print(f'+++++++ projector post processing ', (end-start)*1000)

        return depth_proj 
