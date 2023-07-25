import os.path
import pdb

import torch
from train import NeRFSystem
from opt import get_opts
from utils import slim_ckpt

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

hparams = get_opts()
onnx_name = os.path.join(os.path.dirname(hparams.ckpt_path), 'unet_' + hparams.exp_name)
recon_only = True

if 'Synthetic' in hparams.root_dir:
    dataset_type = 'Synthetic_NeRF'
elif 'Tanks' in hparams.root_dir:
    dataset_type = 'TanksAndTemple'
elif 'Blend' in hparams.root_dir:
    dataset_type = 'BlendedMVS'

pdb.set_trace()
## 加载模型
# ckpt_ = slim_ckpt(hparams.ckpt_path,
#                       save_poses=hparams.optimize_ext)
# new_ckpt = f'/work/Users/lisicheng/Code/SteerNeRF/ckpts/nsvf/{hparams.exp_name}/End2End_direct/{hparams.exp_name}.ckpt'
# torch.save(ckpt_, new_ckpt)
# hparams.ckpt_path = new_ckpt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
system = NeRFSystem(hparams)
system.register_buffer('directions', torch.ones(40000,3))
system.register_buffer('test_directions', torch.ones(40000,3))
system.register_buffer('poses', torch.ones(1100,hparams.frame_num,3,4))
checkpoint = torch.load(hparams.ckpt_path)

## 提取出超分模块
srmodel = system.SRmodel.cuda()
sr_dict = srmodel.state_dict()
#pretrain_dict = checkpoint['state_dict']
pretrain_dict = checkpoint
dict_ = {}
for key in pretrain_dict.keys():
    if key.startswith('SRmodel'):
        dict_[key[8:]] = pretrain_dict[key]
sr_dict.update(dict_)
srmodel.load_state_dict(sr_dict)

if recon_only:
    pdb.set_trace()
    model = srmodel.model.cuda()
    ## 虚构输入 a 和 b，以明确输入的 shape
    ## default mode: a = (1,6,800,800); b = (1,3,800,800)
    ## (B, F, H, W) B: batch_size; F: frame_num;
    hw_info = DatasetInfoDict[dataset_type]

    a = torch.randn((1, 6*hparams.frame_num, hw_info['h'], hw_info['w'])).cuda()
    b = torch.randn((1, 3, hw_info['h'], hw_info['w'])).cuda()
    # import pdb
    # pdb.set_trace()
    ## 为输入起名称
    input_name = ['rgb','rgb_up']
    output_name = ['output']
    ## 导出 onnx 模型
    torch.onnx.export(model, (a,b), onnx_name + '.onnx' if recon_only else onnx_name + '.onnx',
         input_names=input_name, output_names=output_name, export_params=True, verbose=False, opset_version=16)
else:
    system.setup(0)
    dataloader = system.test_dataloader()
    input = next(iter(dataloader))

    rgb_low = torch.randn((1, system.hparams.frame_num, 3, system.test_dataset.low_res_h, system.test_dataset.low_res_w)).cuda()
    if system.hparams.feature_training:
        feat_low = torch.randn((1, system.hparams.frame_num, 3, system.test_dataset.low_res_h, system.test_dataset.low_res_w)).cuda()
    depth = torch.randn((1, system.hparams.frame_num, 1, system.test_dataset.low_res_h, system.test_dataset.low_res_w)).cuda()
    opacity = torch.randn((1, system.hparams.frame_num, 1, system.test_dataset.low_res_h, system.test_dataset.low_res_w)).cuda()


    if system.hparams.feature_training:
        input_name = ['rgb','depth','pose','offset','feat','K_mat']
        output_name = ['output']
        torch.onnx.export(srmodel, (rgb_low, depth, input['pose'].cuda(), input['offset'].cuda(),  feat_low, input['low_res_K'].cuda()), 
            onnx_name + '_re.onnx' if recon_only else onnx_name + '.onnx', input_names=input_name, output_names=output_name, export_params=True, verbose=False, opset_version=16)
    else: # multiple_frames
        input_name = ['rgb','depth','pose', 'offset', 'K', 'mask', 'feat']
        output_name = ['output']
        torch.onnx.export(srmodel, (rgb_low,depth,input['pose'],input['offset'],input['low_res_K'],None,None), 
            onnx_name + '_re.onnx' if recon_only else onnx_name + '.onnx', input_names=input_name, output_names=output_name, export_params=True, verbose=False, opset_version=16)
