import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import warp, make_grid, spectral_norm
from submodels import Generative_Encoder, Generative_Decoder, Evolution_Network,Noise_Projector
from torchvision.models import resnet50




class NowcastNet(nn.Module):
    def __init__(self, configs):
        super(NowcastNet, self).__init__()
        self.configs = configs
        self.pred_length = configs.pred_length
        self.evo_net = Evolution_Network(self.configs.input_length, self.pred_length, base_c=32)
        self.gen_enc = Generative_Encoder(2*self.configs.total_length, base_c=self.configs.ngf)
        self.gen_dec = Generative_Decoder(self.configs)
        self.proj = Noise_Projector(self.configs.ngf, configs)
        self.conv_merge = nn.Conv3d(in_channels=3, out_channels=1, kernel_size=1)
        sample_tensor = torch.zeros(1, 1, self.configs.img_height, self.configs.img_width)
        self.grid = make_grid(sample_tensor)


    def forward(self, all_frames):
        # all_frames = all_frames[:, :, :, :, :1]
        all_frames = all_frames.permute(0,4,2,3,1) #bs, t, h, w, c -> bs, c, h, w, t
        all_frames = self.conv_merge(all_frames)
        all_frames = all_frames.permute(0,4,2,3,1) # bs, t, h, w, 1
        frames = all_frames.permute(0, 1, 4, 2, 3) # bs, t, 1, h, w
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        # Input Frames
        input_frames = frames[:, :self.configs.input_length]
        input_frames = input_frames.reshape(batch, self.configs.input_length, height, width) # bs, t, h, w

        # Evolution Network
        intensity, motion = self.evo_net(input_frames)
        motion_ = motion.reshape(batch, self.pred_length, 2, height, width)
        intensity_ = intensity.reshape(batch, self.pred_length, 1, height, width)
        series = []
        last_frames = all_frames[:, (self.configs.input_length - 1):self.configs.input_length, :, :, 0]
        grid = self.grid.repeat(batch, 1, 1, 1)
        for i in range(self.pred_length):
            # last_frames = warp(last_frames, motion_[:, i], grid.cuda(), mode="nearest", padding_mode="border")
            last_frames = warp(last_frames, motion_[:, i], grid, mode="nearest", padding_mode="border")
            last_frames = last_frames + intensity_[:, i]
            series.append(last_frames)
        evo_result = torch.cat(series, dim=1)

        evo_result = evo_result/128
        
        # Generative Network
        evo_feature = self.gen_enc(torch.cat([input_frames, evo_result], dim=1))
        # evo_feature = self.gen_enc(input_frames)

        noise = torch.randn(batch, self.configs.ngf, height // 32, width // 32).cuda()
        noise_feature = self.proj(noise).reshape(batch, -1, 4, 4, 8, 8).permute(0, 1, 4, 5, 2, 3).reshape(batch, -1, height // 8, width // 8)

        feature = torch.cat([evo_feature, noise_feature], dim=1)
        gen_result = self.gen_dec(feature, evo_result)

        return gen_result.unsqueeze(-1)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Load a pre-trained ResNet50 model
        self.resnet = resnet50(pretrained=False)

        # Modify the first layer to accept single-channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the last layer to output a single probability value
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze().unsqueeze(1) # Reshape the input tensor
        return self.resnet(x)
    
class Temporal_Discriminator(nn.Module):
    def __init__(self, configs):
        super(Temporal_Discriminator, self).__init__()
        """
        假设:
          configs.input_length -> T_in
          configs.evo_ic       -> T_out
        """
        self.T_in = configs.input_length * configs.input_channel
        self.T_out = configs.pred_length

        # -------------------
        # 第一条路径: 2D 卷积
        # in_channels = T_in, out_channels = out_channels_1
        # -------------------
        self.out_channels_1 = 64
        self.conv1 = spectral_norm(
            nn.Conv2d(
                in_channels=self.T_in + self.T_out,
                out_channels=self.out_channels_1,
                kernel_size=9,
                stride=2,
                padding=9 // 2
            )
        )

        # -------------------
        # 第二条路径: 3D 卷积
        # in_channels = 1, out_channels=4, kernel=(T_in,9,9), stride=(1,2,2)
        # 最终会展平成 out_channels_2 = 4*(some_time_size)
        # 如果 time 方向被完全卷积到 1, 则 out_channels_2=4
        # -------------------
        self.conv2 = spectral_norm(
            nn.Conv3d(
                in_channels=1,
                out_channels=4,
                kernel_size=(self.T_in, 9, 9),
                stride=(1, 2, 2),
                padding=(0, 9 // 2, 9 // 2)
            )
        )
        # 根据需要, 您可以根据 padding/stride 计算最后的实际通道
        # 这里示例把它称为 out_channels_2:
        self.out_channels_2 = 4 * (self.T_out+1)  # 如果 time 卷到 1, 即 4*(1)

        # -------------------
        # 第三条路径: 3D 卷积
        # in_channels=1, out_channels=8, kernel=(T_out,9,9), stride=(1,2,2)
        # 最终展平 => out_channels_3 = 8*(some_time_size)
        # -------------------
        self.conv3 = spectral_norm(
            nn.Conv3d(
                in_channels=1,
                out_channels=8,
                kernel_size=(self.T_out, 9, 9),
                stride=(1, 2, 2),
                padding=(0, 9 // 2, 9 // 2)
            )
        )
        self.out_channels_3 = 8* (self.T_in+1)  # 同理

        # 最终拼接通道数
        concat_channels = self.out_channels_1 + self.out_channels_2 + self.out_channels_3

        # ----------------------------------------------------------------
        # 下面构造 4 个 LBlock
        # 第1次: in_channel=concat_channels -> out_channel=128, stride=2
        # 第2次: 128 -> 256, stride=2
        # 第3次: 256 -> 512, stride=2
        # 第4次: 512 -> 512, stride=1 (不再减 spatial 尺寸)
        # ----------------------------------------------------------------
        self.lblock1 = DoubleConv(concat_channels, 128, stride=2)
        self.lblock2 = DoubleConv(128, 256, stride=2)
        self.lblock3 = DoubleConv(256, 512, stride=2)
        self.lblock4 = DoubleConv(512, 512, stride=1)

        self.BN = nn.BatchNorm2d(512)
        self.leaky_relu = nn.LeakyReLU(negative_slope=1e-2)
        self.conv_out = spectral_norm(nn.Conv2d(512, 1, kernel_size=3, padding=1, stride=1))


    def forward(self, x_bar, x_input):
        """
        x_input.shape = [B, T, C, H, W]
        x_bar.shape   = [B, T2, C, H, W]
        其中:
          - 在时间维 x_input 拿前 self.T_in 帧
          - x_bar 拿前 self.T_out 帧
          - C=1 或多通道时, 这里只示例取第 0 通道
        """
        x = torch.cat([x_bar, x_input], dim=1)
        B, T, C, H, W = x_input.shape
        # 1) 取前 T_in 帧并做 2D 卷积
        #    conv1 期望输入是 [B, T_in, H, W]
        x_in2d = x[:, :,0]  # -> [B, T_in, H, W]
        feat1 = self.conv1(x_in2d)              # -> [B, out_channels_1, H/2, W/2]

        # 2) 再对同样的 x_in2d 做 3D 卷积
        #    conv2 期望 [B, 1, T_in, H, W]
        x_in3d = x_in2d.unsqueeze(1)            # -> [B, 1, T_in, H, W]
        feat2 = self.conv2(x_in3d)             # -> [B, 4, 1, H/2, W/2] (若time维度卷积到1)
        feat2 = feat2.reshape(B, self.out_channels_2, H//2, W//2)              # -> [B, 4, H/2, W/2]

        # 3) 对 x_bar 做 3D 卷积
        feat3 = self.conv3(x_in3d)            # -> [B, 8, 1, H/2, W/2]
        feat3 = feat3.reshape(B, self.out_channels_3, H//2, W//2)              # -> [B, 8, H/2, W/2]

        # 4) 拼接通道 => [B, out_channels_1 + out_channels_2 + out_channels_3, H/2, W/2]
        feats = torch.cat([feat1, feat2, feat3], dim=1)

        # 5) 依次通过 4 个 LBlock
        #    尺寸变化:
        #      - lblock1: -> [B, 128, H/4, W/4]
        #      - lblock2: -> [B, 256, H/8, W/8]
        #      - lblock3: -> [B, 512, H/16, W/16]
        #      - lblock4: -> [B, 512, H/16, W/16]
        out = self.lblock1(feats)
        out = self.lblock2(out)
        out = self.lblock3(out)
        out = self.lblock4(out)

        # 这里您可以根据需求返回判别器的“打分”或者特征
        # 若需要一个标量或向量, 还可以在最后做一次全局池化/全连接等
        out = self.BN(out)
        out = self.leaky_relu(out)
        out = self.conv_out(out)
        return out
    
class DoubleConv(nn.Module):
    def   __init__(self, in_channels, out_channels, kernel=3, stride=2, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, padding=kernel//2, stride=stride)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=kernel, padding=kernel//2, stride=1)),
        )
        self.single_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel//2, stride=stride))
        )

    def forward(self, x):
        shortcut = self.single_conv(x)
        x = self.double_conv(x)
        x = x + shortcut
        return x