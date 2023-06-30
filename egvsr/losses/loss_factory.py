from absl.logging import info
from torch.nn.modules.loss import _Loss

from egvsr.losses.image_loss import (
    ImageSRLoss,
    RandomScaleImageSRLoss,
    RSSRLoss,
    RSSREventPolarityLoss,
    RSSRALPXEventPolarityLoss,
)


def get_single_loss(config):
    if config.NAME == "ImageSRLoss":
        return ImageSRLoss()
    elif config.NAME == "RSSRLoss":
        return RSSRLoss()
    elif config.NAME == "RSSRALPXEventPolarityLoss":
        return RSSRALPXEventPolarityLoss()
    elif config.NAME == "RSSREventPolarityLoss":
        return RSSREventPolarityLoss()
    else:
        raise ValueError(f"Unknown config: {config}")


class MixedLoss(_Loss):
    def __init__(self, configs):
        super(MixedLoss, self).__init__()
        self.loss = []
        self.weight = []
        self.criterion = []
        for item in configs:
            self.loss.append(item.NAME)
            self.weight.append(item.WEIGHT)
            self.criterion.append(get_single_loss(item))
        info(f"Init Mixed Loss: {configs}")

    def forward(self, outputs, sr, lr, lr_event, hr_event):
        name_to_loss = []
        total = 0
        for n, w, fun in zip(self.loss, self.weight, self.criterion):
            tmp = fun(outputs, sr, lr, lr_event, hr_event)
            name_to_loss.append((n, tmp))
            total = total + tmp * w
        return total, name_to_loss
