from absl.logging import info
from torch import nn

from egvsr.losses.psnr import EGVSRPSNR, RandomScalePSNR, RSSRPSNR
from egvsr.losses.ssim import EGVSRSSIM, RandomScaleSSIM, RSSRSSIM


def get_single_metric(config):
    if config.NAME == "SSIM":
        return EGVSRSSIM()
    elif config.NAME == "PSNR":
        return EGVSRPSNR()
    elif config.NAME == "RSSRPSNR":
        return RSSRPSNR()
    elif config.NAME == "RSSRSSIM":
        return RSSRSSIM()
    elif config.NAME.startswith("RandomScaleSSIM"):
        scale = int(config.NAME.split("-")[1])
        return RandomScaleSSIM(scale)
    elif config.NAME.startswith("RandomScalePSNR"):
        scale = int(config.NAME.split("-")[1])
        return RandomScalePSNR(scale)
    else:
        raise ValueError(f"Unknown config: {config}")


class MixedMetric(nn.Module):
    def __init__(self, configs):
        super(MixedMetric, self).__init__()
        self.metric = []
        self.eval = []
        for config in configs:
            self.metric.append(config.NAME)
            self.eval.append(get_single_metric(config))
        info(f"Init Mixed Metric: {configs}")

    def forward(self, outputs, sr, lr, lr_event, hr_event):
        r = []
        for m, e in zip(self.metric, self.eval):
            r.append((m, e(outputs, sr, lr, lr_event, hr_event)))
        return r
