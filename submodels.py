import torch.nn.functional as F
from blocks import *
from utils import spectral_norm

class Evolution_Network(nn.Module):
    def __init__(self, n_channels, n_classes, base_c=64, bilinear=True):
        super(Evolution_Network, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        base_c = base_c
        self.inc = DoubleConv(n_channels, base_c)
        self.down1 = Down(base_c * 1, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c * 1, bilinear)
        self.outc = OutConv(base_c * 1, n_classes)
        self.gamma = nn.Parameter(torch.zeros(1, n_classes, 1, 1), requires_grad=True)

        self.up1_v = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2_v = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3_v = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4_v = Up(base_c * 2, base_c * 1, bilinear)
        self.outc_v = OutConv(base_c * 1, n_classes * 2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x) * self.gamma

        v = self.up1_v(x5, x4)
        v = self.up2_v(v, x3)
        v = self.up3_v(v, x2)
        v = self.up4_v(v, x1)
        v = self.outc_v(v)
        return x, v


class Generative_Encoder(nn.Module):
    def __init__(self, n_channels, base_c=64):
        super(Generative_Encoder, self).__init__()
        base_c = base_c
        self.inc = DoubleConv(n_channels, base_c, kernel=3)
        self.down1 = Down(base_c * 1, base_c * 2, 3)
        self.down2 = Down(base_c * 2, base_c * 4, 3)
        self.down3 = Down(base_c * 4, base_c * 8, 3)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x

class Generative_Decoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        ic = opt.ic_feature
        self.fc = nn.Conv2d(ic, 8 * nf, 3, padding=1)

        self.head_0 = GenBlock(8 * nf, 8 * nf, opt)

        self.G_middle_0 = GenBlock(8 * nf, 4 * nf, opt, double_conv=True)
        self.G_middle_1 = GenBlock(4 * nf, 4 * nf, opt, double_conv=True)

        self.up_0 = GenBlock(4 * nf, 2 * nf, opt)

        self.up_1 = GenBlock(2 * nf, 1 * nf, opt, double_conv=True)
        self.up_2 = GenBlock(1 * nf, 1 * nf, opt, double_conv=True)

        final_nc = nf * 1
        self.evo_pooling = nn.AvgPool2d(8, stride=8)
        self.evo_conv = nn.Conv2d(opt.input_length,2*opt.input_length, kernel_size=1, stride=1, padding=0)
        self.conv_img = nn.Conv2d(final_nc, self.opt.gen_oc, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
    
    def forward(self, x, evo):
        x = self.fc(x)
        evo = self.evo_conv(self.evo_pooling(evo))
        x = self.head_0(x, evo)
        x = self.up(x)
        x = self.G_middle_0(x, evo)
        x = self.G_middle_1(x, evo)
        x = self.up(x)
        x = self.up_0(x, evo)
        x = self.up(x)
        x = self.up_1(x, evo)
        x = self.up_2(x, evo)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        return x


class Noise_Projector(nn.Module):
    def __init__(self, input_length, configs):
        super(Noise_Projector, self).__init__()
        self.input_length = input_length
        self.conv_first = spectral_norm(nn.Conv2d(self.input_length, self.input_length * 2, kernel_size=3, padding=1))
        self.L1 = ProjBlock(self.input_length * 2, self.input_length * 4)
        self.L2 = ProjBlock(self.input_length * 4, self.input_length * 8)
        self.L3 = ProjBlock(self.input_length * 8, self.input_length * 16)
        self.L4 = ProjBlock(self.input_length * 16, self.input_length * 32)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)

        return x


class ProjBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ProjBlock, self).__init__()
        self.one_conv = spectral_norm(nn.Conv2d(in_channel, out_channel-in_channel, kernel_size=1, padding=0))
        self.double_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
        )

    def forward(self, x):
        x1 = torch.cat([x, self.one_conv(x)], dim=1)
        x2 = self.double_conv(x)
        output = x1 + x2
        return output