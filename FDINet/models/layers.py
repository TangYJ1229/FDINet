import torch
from torch import Tensor, reshape, stack
from torch import nn
from models.DWT_IDWT.DWT_IDWT_layer import *
import torch.nn.functional as F
import math

def conv_shape(x):
    B, N, C = x.shape
    H = int(math.sqrt(N))
    W = int(math.sqrt(N))
    x = x.reshape(B, C, H, W).contiguous()
    return x


class CBA3x3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.cba = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.cba(x)


class CBA1x1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.cba = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.cba(x)


class MixConv2d(nn.Module):
    def __init__(self, ch_out):
        super().__init__()
        self._convmix = nn.Sequential(
            nn.Conv2d(ch_out*2, ch_out, 3, groups=ch_out, padding=1),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(),
        )

    def forward(self, x, y):
        # Packing the tensors and interleaving the channels:
        mixed = torch.stack((x, y), dim=2)
        mixed = torch.reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))

        # Mixing:
        return self._convmix(mixed)


class MixConv3d(nn.Module):
    def __init__(self, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_out, ch_out, (2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(),
        )

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(2)
        x2 = x2.unsqueeze(2)

        x = torch.cat([x1, x2], dim=2)
        x = self.conv(x).squeeze(2)
        return x


class TemporalConv3d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.temporal_cba = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, (2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(2)
        x2 = x2.unsqueeze(2)
        temporal = torch.cat([x1, x2], dim=2)
        temporal = self.temporal_cba(temporal).squeeze(2)
        return temporal


class Change_Information_Extracter(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        #information
        self.mix3d = MixConv3d(in_channel)
        #change
        self.relu = nn.ReLU()
        self.conv1 = CBA1x1(in_channel, in_channel)
        self.conv2 = CBA1x1(in_channel, in_channel)

        #cat
        self.cat = CBA3x3(in_channel*2, in_channel)

    def forward(self, x1, x2):
        information = self.mix3d(x1, x2)

        banch1 = self.conv1(self.relu(x1 - x2))
        banch2 = self.conv2(self.relu(x2 - x1))
        change = torch.cat([banch1, banch2], dim=1)
        change = self.cat(change)

        x = information + change
        return x

class HH_Interation(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv = CBA1x1(in_channel, in_channel)

    def forward(self, HH1, HH2):
        HH = torch.abs(HH1 - HH2)
        HH = self.conv(HH)
        return HH


class LH_Interaction(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.temporal = TemporalConv3d(in_channel, in_channel)

    def forward(self, LH1, LH2):
        x = self.temporal(LH1, LH2)
        return x

class HL_Interaction(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.temporal = TemporalConv3d(in_channel, in_channel)

    def forward(self, HL1, HL2):
        x = self.temporal(HL1, HL2)
        return x


class LL_Interation(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.mix3d = MixConv3d(in_channel)
        # change
        self.relu = nn.ReLU()
        self.conv1 = CBA1x1(in_channel, in_channel)
        self.conv2 = CBA1x1(in_channel, in_channel)

        # cat
        self.cat = CBA3x3(in_channel * 2, in_channel)

    def forward(self, LL1, LL2):
        information = self.mix3d(LL1, LL2)

        banch1 = self.conv1(self.relu(LL1 - LL2))
        banch2 = self.conv2(self.relu(LL2 - LL1))
        change = torch.cat([banch1, banch2], dim=1)
        change = self.cat(change)

        x = information + change
        return x


class Wavelet_Skip(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.dwt = DWT_2D('haar')

        self.LL = LL_Interation(in_channel)
        self.LH = LH_Interaction(in_channel)
        self.HL = HL_Interaction(in_channel)
        self.HH = HH_Interation(in_channel)

        self.idwt = IDWT_2D('haar')

    def forward(self, x1_downsample, x2_downsample):
        ll1, lh1, hl1, hh1 = self.dwt(x1_downsample)
        ll2, lh2, hl2, hh2 = self.dwt(x2_downsample)

        LL = self.LL(ll1, ll2)
        LH = self.LH(lh1, lh2)
        HL = self.HL(hl1, hl2)
        HH = self.HH(hh1, hh2)

        x = self.idwt(LL, LH, HL, HH)
        return x


class Feature_Aggreation_Module(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.origin = nn.Sequential(
            CBA3x3(channel[0], channel[1]),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

        self.large = CBA3x3(channel[1], channel[1])

        self.verylarge = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            CBA3x3(channel[2], channel[1])
        )

        self.conv = CBA1x1(channel[1], channel[1])
        self.cat = CBA3x3(channel[1] * 3, channel[1])

    def forward(self, x, l, vl):
        x = self.origin(x)

        res = l
        l = self.large(l)

        vl = self.verylarge(vl)

        x = torch.cat([x, l, vl], dim=1)
        x = self.cat(x)

        x = x + res
        x = self.conv(x)
        return x


class First_FAM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.origin = nn.Sequential(
            CBA3x3(channel[0], channel[1]),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

        self.large = CBA3x3(channel[1], channel[1])

        self.conv = CBA1x1(channel[1], channel[1])
        self.cat = CBA3x3(channel[1] * 2, channel[1])

    def forward(self, x, l):
        x = self.origin(x)

        res = l
        l = self.large(l)

        x = torch.cat([x, l], dim=1)
        x = self.cat(x)

        x = x + res
        x = self.conv(x)
        return x

class Interation1(nn.Module):#消融  cat
    def __init__(self, channel):
        super().__init__()
        self.cat = nn.Sequential(
            CBA3x3(channel * 2, channel),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.cat(x)
        return x


class Interation2(nn.Module):#消融  temporal
    def __init__(self, channel):
        super().__init__()
        self.temporal = TemporalConv3d(channel, channel)

    def forward(self, x1, x2):
        x = self.temporal(x1, x2)
        return x

class Interation3(nn.Module):#消融   3d
    def __init__(self, channel):
        super().__init__()
        self.mix3d = MixConv3d(channel)

    def forward(self, x1, x2):
        x = self.mix3d(x1, x2)
        return x


class upsample(nn.Module):#消融2
    def __init__(self, channel):
        super().__init__()
        self.origin = nn.Sequential(
            CBA3x3(channel[0], channel[1]),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )
        self.cat = CBA3x3(channel[1] * 2, channel[1])

    def forward(self, x, l):
        x = self.origin(x)
        x = torch.cat([x, l], dim=1)
        x = self.cat(x)
        return x


class Classifier(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.cba = nn.Sequential(
            nn.ConvTranspose2d(in_channel, in_channel, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channel, int(in_channel/2), 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(int(in_channel/2), out_channel, 1, 1, 0),
        )
    def forward(self, x):
        x = self.cba(x)
        return x