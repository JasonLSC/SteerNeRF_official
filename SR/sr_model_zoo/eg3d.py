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

class EG3D(nn.Module):
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

        self.rgb_upsample = torch.nn.UpsamplingBilinear2d(scale_factor=4)


    def forward(self, rgbd):
        rgb = rgbd[:, 0:3, :, :]

        feat_1 = self.conv_up_1(rgbd)
        y_1 = self.feat2rgb_1(feat_1)
        rgb_up_1 = self.rgb_upsample_1(rgb)
        rgb_1 = y_1 + rgb_up_1

        feat_2 = self.conv_up_2(feat_1)
        y_2 = self.feat2rgb_2(feat_2)
        rgb_up_2 = self.rgb_upsample_2(rgb_1)
        rgb_2 = y_2 + rgb_up_2

        return rgb_2

if __name__ == "__main__":
    Model = EG3D(4, 3, upsample_kernel=3)
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