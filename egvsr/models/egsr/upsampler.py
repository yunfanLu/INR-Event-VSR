import math

from torch import nn


class Upsampler(nn.Module):
    def __init__(self, scale, n_feats):
        super(Upsampler, self).__init__()
        self.m = nn.ModuleList()
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                self.m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1))
                self.m.append(nn.PixelShuffle(2))
        elif scale == 3:
            self.m.append(nn.Conv2d(n_feats, 9 * n_feats, 3, padding=1))
            self.m.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError

    def forward(self, x):
        for m in self.m:
            x = m(x)
        return x
