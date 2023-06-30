import torch
import torch.nn.functional as F
from torch import nn

from egvsr.models.egsr.scale import Weight


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResidualUnits(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(ResidualUnits, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.conv = nn.Conv2d(
            in_channels,
            growth_rate,
            kernel_size=kernel_size,
            padding=kernel_size >> 1,
            stride=1,
        )
        self.conv1 = nn.Conv2d(
            growth_rate,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size >> 1,
            stride=1,
        )
        self.relu = nn.PReLU(growth_rate)
        self.weight1 = Weight(1)
        self.weight2 = Weight(1)

    def forward(self, x):
        x1 = self.conv1(self.relu(self.conv(x)))
        output = self.weight1(x) + self.weight2(x1)
        return output


class ConvRelu(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=1,
    ):
        super(ConvRelu, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class AdaptiveResidualFeatureBlock(nn.Module):
    def __init__(self, n_feats):
        super(AdaptiveResidualFeatureBlock, self).__init__()
        self.layer1 = ResidualUnits(n_feats, n_feats // 2, 3)
        self.layer2 = ResidualUnits(n_feats, n_feats // 2, 3)
        self.layer4 = ConvRelu(n_feats, n_feats, 3, 1, 1)
        self.alise = ConvRelu(2 * n_feats, n_feats, 1, 1, 0)
        self.atten = CALayer(n_feats)
        self.weight1 = Weight(1)
        self.weight2 = Weight(1)
        self.weight3 = Weight(1)
        self.weight4 = Weight(1)
        self.weight5 = Weight(1)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = torch.cat([self.weight2(x2), self.weight3(x1)], 1)
        x4 = self.alise(x3)
        x5 = self.atten(x4)
        x6 = self.layer4(x5)
        return self.weight4(x) + self.weight5(x6)


class HighPreservingBlock(nn.Module):
    def __init__(self, n_feats):
        super(HighPreservingBlock, self).__init__()
        self.encoder = AdaptiveResidualFeatureBlock(n_feats)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.decoder_low = AdaptiveResidualFeatureBlock(n_feats)
        self.decoder_high = AdaptiveResidualFeatureBlock(n_feats)
        self.alise2 = ConvRelu(2 * n_feats, n_feats, 1, 1, 0)
        self.att = CALayer(n_feats)
        self.alise = AdaptiveResidualFeatureBlock(n_feats)

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.down(x1)
        high = x1 - F.interpolate(x2, size=x.size()[-2:], mode="bilinear", align_corners=True)
        for i in range(5):
            x2 = self.decoder_low(x2)
        x3 = x2
        x4 = F.interpolate(x3, size=x.size()[-2:], mode="bilinear", align_corners=True)
        high1 = self.decoder_high(high)
        x5 = self.alise2(torch.cat([x4, high1], dim=1))
        x6 = self.att(x5)
        x7 = self.alise(x6) + x
        return x7


class LightWeightCNNBackbone(nn.Module):
    def __init__(self, in_channels, depth):
        super(LightWeightCNNBackbone, self).__init__()
        self.depth = depth
        self.hpbs = nn.ModuleList()
        for i in range(depth):
            self.hpbs.append(HighPreservingBlock(in_channels))
        self.reduce = nn.Conv2d(depth * in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        xs = []
        for i in range(self.depth):
            xs.append(self.hpbs[i](x))
        out = torch.cat(xs, dim=1)
        out = self.reduce(out)
        return out
