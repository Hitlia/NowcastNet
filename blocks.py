import re
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

class DoubleConv(nn.Module):

    def   __init__(self, in_channels, out_channels, kernel=3, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, padding=kernel//2)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=kernel, padding=kernel//2)),
        )
        self.single_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2))
        )

    def forward(self, x):
        shortcut = self.single_conv(x)
        x = self.double_conv(x)
        x = x + shortcut
        return x

class Down(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True, kernel=3):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, kernel=kernel, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up_S(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True, kernel=3):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, kernel=kernel, mid_channels=in_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class GenBlock(nn.Module):
    def __init__(self, fin, fout, opt, use_se=False, dilation=1, double_conv=False):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.opt = opt
        self.double_conv = double_conv

        self.pad = nn.ReflectionPad2d(dilation)
        self.conv_0 = spectral_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=0, dilation=dilation))
        self.conv_1 = spectral_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=0, dilation=dilation))

        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        ic = opt.evo_ic
        
        self.norm_0 = SPADE(fin, ic)
        self.norm_1 = SPADE(fmiddle, ic)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, ic)

    def forward(self, x, evo):
        x_s = self.shortcut(x, evo)
        dx = self.conv_0(self.pad(self.actvn(self.norm_0(x, evo))))
        if self.double_conv:
            dx = self.conv_1(self.pad(self.actvn(self.norm_1(dx, evo))))

        out = x_s + dx

        return out

    def shortcut(self, x, evo):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, evo))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
    
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        # label_nc是输出帧数，从label_nc卷积到nhidden再到norm_nc

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 64
        ks = 3
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=0),
            nn.ReLU()
        )
        self.pad = nn.ReflectionPad2d(pw)
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0)

    def forward(self, x, evo):

        normalized = self.param_free_norm(x)
        evo = F.adaptive_avg_pool2d(evo, output_size=x.size()[2:])

        actv = self.mlp_shared(evo)

        gamma = self.mlp_gamma(self.pad(actv))
        beta = self.mlp_beta(self.pad(actv))

        out = normalized * (1 + gamma) + beta

        return out