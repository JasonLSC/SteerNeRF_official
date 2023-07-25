import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class PixelShufflePack(nn.Module):
    """Pixel Shuffle upsample layer.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.
    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        """Initialize weights for PixelShufflePack."""
        default_init_weights(self, 1)

    def forward(self, x):
        """Forward function for PixelShufflePack.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x

class UpsampleModule(nn.Sequential):
    """Upsample module used in EDSR.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        mid_channels (int): Channel number of intermediate features.
    """

    def __init__(self, scale, mid_channels):
        modules = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                modules.append(
                    PixelShufflePack(
                        mid_channels, mid_channels, 2, upsample_kernel=3))
        elif scale == 3:
            modules.append(
                PixelShufflePack(
                    mid_channels, mid_channels, scale, upsample_kernel=3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')

        super().__init__(*modules)

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

class styleganSR(nn.Module):
    def __init__(self, in_channels, out_channels,
                 scale_factor=4, upsample_kernel=3):
        super().__init__()

        self.conv_up_1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            PixelShufflePack(128, 128, scale_factor=2, upsample_kernel=upsample_kernel),
            # nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            # nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(inplace=True),
        )

        self.feat2rgb_1 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        self.rgb_upsample_1 = torch.nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv_up_2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            PixelShufflePack(64, 64, scale_factor=2, upsample_kernel=upsample_kernel),
            # nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            # nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(inplace=True),
        )

        self.feat2rgb_2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.rgb_upsample_2 = torch.nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv_end = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        )

        # self.rgb_upsample = torch.nn.UpsamplingNearest2d(scale_factor=4)
        self.rgb_upsample = torch.nn.UpsamplingBilinear2d(scale_factor=4)

        # refactor
        # block_num = 3
        # mid_channel = [128, 64, 32]
        # self.conv_first = nn.Conv2d(in_channels, mid_channel[0], 3, padding=1) # bias?
        #
        # self.conv_blocks_1 = nn.ModuleList()
        # self.conv_blocks_2 = nn.ModuleList()
        # self.conv_blocks_3 = nn.ModuleList()
        # for _ in range(block_num):
        #     self.conv_blocks_1.append(ResidualConvBlock(mid_channel[0], ks=3, act_name='prelu'))
        #     self.conv_blocks_2.append(ResidualConvBlock(mid_channel[1], ks=3, act_name='prelu'))
        #     self.conv_blocks_3.append(ResidualConvBlock(mid_channel[2], ks=3, act_name='prelu'))
        #
        # self.feat_upsampling_1 = PixelShufflePack(mid_channel[0], mid_channel[1], upsample_kernel=3, scale_factor=2)
        # self.feat_upsampling_2 = PixelShufflePack(mid_channel[1], mid_channel[2], upsample_kernel=3, scale_factor=2)
        #
        # self.conv_last = nn.Conv2d(mid_channel[-1], 3, 1)
        # self.last_act_func = nn.ReLU()

    def forward(self, rgbd):
        # draft
        rgb = rgbd[:, 0:3, :, :]
        up_rgb = self.rgb_upsample(rgb)
        return self.conv_end(self.conv_up_2(self.conv_up_1(rgbd))) + up_rgb

        # rgb = rgbd[:, 0:3, :, :]
        # up_rgb = self.rgb_upsample(rgb)
        #
        # feat_1 = self.conv_first(rgbd)
        # for i, module in enumerate(self.conv_blocks_1):
        #     feat_1 = module(feat_1)
        # feat_2 = self.feat_upsampling_1(feat_1)
        #
        # for i, module in enumerate(self.conv_blocks_2):
        #     feat_2 = module(feat_2)
        # feat_3 = self.feat_upsampling_2(feat_2)
        #
        # for i, module in enumerate(self.conv_blocks_3):
        #     feat_3 = module(feat_3)
        # y = self.conv_last(feat_3) # 1x1 conv
        #
        # out = y + up_rgb
        #
        # return out

class styleganSR_res(nn.Module):
    def __init__(self, in_channels, out_channels,
                 scale_factor=4, upsample_kernel=3):
        super().__init__()
        self.rgb_upsample = torch.nn.UpsamplingBilinear2d(scale_factor=4)

        # refactor
        block_num = 4
        mid_channel = [128, 64, 32]
        self.conv_first = nn.Conv2d(in_channels, mid_channel[0], 3, padding=1) # bias?

        self.conv_blocks_1 = nn.ModuleList()
        self.conv_blocks_2 = nn.ModuleList()
        self.conv_blocks_3 = nn.ModuleList()
        for _ in range(block_num):
            self.conv_blocks_1.append(ResidualConvBlock(mid_channel[0], ks=3, act_name='relu'))
            self.conv_blocks_2.append(ResidualConvBlock(mid_channel[1], ks=3, act_name='relu'))
            self.conv_blocks_3.append(ResidualConvBlock(mid_channel[2], ks=3, act_name='relu'))

        self.feat_upsampling_1 = PixelShufflePack(mid_channel[0], mid_channel[1], upsample_kernel=3, scale_factor=2)
        self.feat_upsampling_2 = PixelShufflePack(mid_channel[1], mid_channel[2], upsample_kernel=3, scale_factor=2)

        self.conv_last = nn.Conv2d(mid_channel[-1], 3, 1)
        self.last_act_func = nn.ReLU()

    def forward(self, rgbd):
        rgb = rgbd[:, 0:3, :, :]
        up_rgb = self.rgb_upsample(rgb)

        feat_1 = self.conv_first(rgbd)
        for i, module in enumerate(self.conv_blocks_1):
            feat_1 = module(feat_1)
        feat_2 = self.feat_upsampling_1(feat_1)

        for i, module in enumerate(self.conv_blocks_2):
            feat_2 = module(feat_2)
        feat_3 = self.feat_upsampling_2(feat_2)

        for i, module in enumerate(self.conv_blocks_3):
            feat_3 = module(feat_3)
        y = self.conv_last(feat_3) # 1x1 conv

        y = y + up_rgb
        y = self.last_act_func(y)
        return y


if __name__ == "__main__":
    Model = styleganSR(4, 3, upsample_kernel=3)
    trainable_params = sum(
        p.numel() for p in Model.parameters() if p.requires_grad
    )
    print(trainable_params)
    for n, p in Model.named_parameters():
        if p.requires_grad:
            print(n)
            print(p.shape)
            print(p.numel())
            print(f"{p.numel()/trainable_params * 100} %")
            print("==========================")