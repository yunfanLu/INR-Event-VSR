import torch
from absl.logging import info, error
from torch import nn

import torch.nn.functional as F


def almost_equal(x, y):
    return abs(x - y) < 1e-6


class _PSNR(nn.Module):
    def __init__(self):
        super(_PSNR, self).__init__()
        self.eps = torch.tensor(1e-10)

        info(f"Init PSNR:")
        info(f"  Note: the psnr max value is {-10 * torch.log10(self.eps)}")

    def forward(self, x, y):
        d = x - y
        mse = torch.mean(d * d) + self.eps
        psnr = -10 * torch.log10(mse)
        return psnr


class EGVSRPSNR(nn.Module):
    def __init__(self):
        super(EGVSRPSNR, self).__init__()
        self.psnr = _PSNR()

    def forward(self, outputs, sr, lr, event):
        psnr = self.psnr(outputs, sr)
        return psnr


class RandomScalePSNR(nn.Module):
    def __init__(self, scale):
        super(RandomScalePSNR, self).__init__()
        self.psnr = _PSNR()
        self.scale = scale
        info(f"RandomScalePSNR: {scale}")

    def forward(self, outputs, sr, lr, event):
        random_scale_sr_image = outputs[self.scale]
        b, c, h, w = random_scale_sr_image.shape
        b_s, n_s, c_s, h_s, w_s = sr.shape
        sr = sr.reshape(b_s, n_s * c_s, h_s, w_s)

        sr = F.interpolate(sr, size=(h, w), mode="bicubic", align_corners=True)
        psnr = self.psnr(sr, random_scale_sr_image)
        return psnr


class RSSRPSNR(nn.Module):
    def __init__(self):
        super(RSSRPSNR, self).__init__()
        self.psnr = _PSNR()

    def forward(self, outputs, sr, lr, lr_event, hr_event):
        if isinstance(outputs, tuple):
            sr_images, event_reconstruction_polarity = outputs
            outputs = sr_images

        b, c, h, w = outputs.shape
        b_s, n_s, c_s, h_s, w_s = sr.shape
        sr = sr.reshape(b_s, n_s * c_s, h_s, w_s)
        sr = F.interpolate(sr, size=(h, w), mode="bicubic", align_corners=True)
        psnr = self.psnr(sr, outputs)
        return psnr
