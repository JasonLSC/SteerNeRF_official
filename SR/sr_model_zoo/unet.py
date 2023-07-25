import math
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sr_backbones_utils import PixelShufflePack

def kaiming_init(module: nn.Module,
                 a: float = 0,
                 mode: str = 'fan_out',
                 nonlinearity: str = 'relu',
                 bias: float = 0,
                 distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def default_init_weights(module, scale=1):
    """Initialize network weights.
    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.
    It has a style of:
    ::
        ---Conv-ReLU-Conv-+-
         |________________|
    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.
        Initialization methods like `kaiming_init` are for VGG-style modules.
        For modules with residual paths, using smaller std is better for
        stability and performance. We empirically use 0.1. See more details in
        "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class ResidualConvBlock(nn.Module):
    def __init__(self, chs, ks=3,
                 act_name='relu', use_residual=True):
        super().__init__()
        self.use_residual = use_residual

        self.conv_1 = nn.Conv2d(chs, chs, ks, padding=(ks-1)//2)
        self.conv_2 = nn.Conv2d(chs, chs, ks, padding=(ks - 1) // 2)

        self.act_func_list = nn.ModuleList()
        for _ in range(2):
            if act_name == 'relu':
                self.act_func_list.append(nn.ReLU())
            elif act_name == 'prelu':
                self.act_func_list.append(nn.PReLU())

    def forward(self, x):
        feat = self.conv_2(self.act_func_list[0](self.conv_1(x)))
        y = feat + x if self.use_residual else feat
        out = self.act_func_list[1](y)

        return out

class ReconstructionNet_v4(nn.Module):
    def __init__(self, frame_num: int = 1, in_channels = 6):
        super(ReconstructionNet_v4, self).__init__()
        assert (frame_num > 0)

        kernel_size = 3
        padding = 1

        self.pooling = nn.MaxPool2d(2)

        self.first_conv = nn.Conv2d(in_channels * frame_num, 64, kernel_size=kernel_size, padding=padding)
        # pdb.set_trace()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            ResidualBlockNoBN(mid_channels=64, res_scale=1),
            nn.ReLU()
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            ResidualBlockNoBN(mid_channels=128, res_scale=1),
            nn.ReLU(),
            ResidualBlockNoBN(mid_channels=128, res_scale=1),
            nn.ReLU(),
            ResidualBlockNoBN(mid_channels=128, res_scale=1),
            nn.ReLU(),
            PixelShufflePack(128, 128, 2, 3)
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # Why bilinear?
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=kernel_size, padding=padding),  # '128+64' Take tensor from skip connect
            nn.ReLU(),
            ResidualBlockNoBN(mid_channels=64, res_scale=1),
            nn.ReLU(),
            PixelShufflePack(64, 64, 2, 3)
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # Why bilinear?
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=kernel_size, padding=padding),  # '64+32' Take tensor from skip connect
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x0):
        x0 = self.first_conv(x0)
        enc_1 = self.encoder1(x0)
        down_1 = self.pooling(enc_1)

        enc_2 = self.encoder2(down_1)
        down_2 = self.pooling(enc_2)

        up2 = self.bottleneck(down_2)

        dec2 = torch.concat([up2, enc_2], dim=1)
        up1 = self.decoder2(dec2)

        dec1 = torch.concat([up1, enc_1], dim=1)
        out = self.decoder1(dec1)

        return out

class ZeroUpsampling(nn.Module):
    def __init__(self, channels: int, upsampling_factor):
        super().__init__()
        self.channels = channels
        self.upsampling_factor = upsampling_factor
        kernel = torch.zeros((channels, 1, upsampling_factor[1], upsampling_factor[0]), dtype=torch.float32, requires_grad=False)
        kernel[:, 0, 0, 0] = 1
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor):
        return F.conv_transpose2d(x, self.kernel, stride=self.upsampling_factor, groups=self.channels)

class UNet_feat_slim_multiframe(nn.Module):
    def __init__(self, in_channels: int, upsampling_factor, frame_num=1):
        super().__init__()
        self.in_channels = in_channels
        self.rec_net = ReconstructionNet_v4(frame_num=frame_num, in_channels=in_channels)

    def forward(self, rgb_feats, up_rgb):
        # input "rgbs" shape: [B, frame_num, C, H, W] // frame_num = 3; C = 6
        # *** only support batch_size=1 for now ***
        # but pytorch api only takes 4 dim input
        # so reshape into [B, frame_num*C, H, W]

        out_rgb = self.rec_net(rgb_feats)
        final_output = up_rgb + out_rgb

        return final_output