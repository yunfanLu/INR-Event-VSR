from logging import info

import torch
from absl.logging import error
from torch.nn.modules.loss import _Loss

import torch.nn.functional as F


class L1CharbonnierLossColor(_Loss):
    def __init__(self):
        super(L1CharbonnierLossColor, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        diff = torch.add(x, -y)
        diff_sq = diff * diff
        diff_sq_color = torch.mean(diff_sq, 1, True)
        error = torch.sqrt(diff_sq_color + self.eps)
        loss = torch.mean(error)
        return loss


class ImageSRLoss(_Loss):
    def __init__(self):
        super(ImageSRLoss, self).__init__()
        self.l1_charbonnier = L1CharbonnierLossColor()

    def forward(self, outputs, sr, lr, event):
        loss = self.l1_charbonnier(sr, outputs)
        return loss


class RandomScaleImageSRLoss(_Loss):
    def __init__(self, scale):
        super(RandomScaleImageSRLoss, self).__init__()
        self.l1_charbonnier = L1CharbonnierLossColor()
        self.scale = scale
        info(f"RandomScaleImageSRLoss: {scale}")

    def forward(self, outputs, sr, lr, event):
        random_scale_sr_image = outputs[self.scale]
        b, c, h, w = random_scale_sr_image.shape
        b_s, n_s, c_s, h_s, w_s = sr.shape
        sr = sr.reshape(b_s, n_s * c_s, h_s, w_s)
        sr = F.interpolate(sr, size=(h, w), mode="bicubic")
        loss = self.l1_charbonnier(sr, random_scale_sr_image)
        return loss


class RSSRLoss(_Loss):
    def __init__(self):
        super(RSSRLoss, self).__init__()
        self.l1_charbonnier = L1CharbonnierLossColor()

    def forward(self, outputs, sr, lr, lr_event, hr_event):
        if isinstance(outputs, tuple):
            sr_images, event_reconstruction_polarity = outputs
            outputs = sr_images
        b, c, h, w = outputs.shape
        b_s, n_s, c_s, h_s, w_s = sr.shape
        sr = sr.reshape(b_s, n_s * c_s, h_s, w_s)
        sr = F.interpolate(sr, size=(h, w), mode="bicubic")
        loss = self.l1_charbonnier(sr, outputs)
        return loss


class RSSREventPolarityLoss(_Loss):
    def __init__(self):
        super(RSSREventPolarityLoss, self).__init__()

    def forward(self, outputs, sr, lr, lr_event, hr_event):
        sr_images, event_reconstruction_polarity = outputs
        b, f, c, h, w = hr_event.shape
        event_polarity_positive = hr_event[:, :, 0, :, :]
        event_polarity_negative = hr_event[:, :, 1, :, :]
        event_polarity = event_polarity_positive - event_polarity_negative
        common = event_reconstruction_polarity * event_polarity
        common = -1.0 * common
        common = torch.relu(common)
        loss = torch.mean(common)
        return loss


class RSSRALPXEventPolarityLoss(_Loss):
    def __init__(self):
        super(RSSRALPXEventPolarityLoss, self).__init__()

    def forward(self, outputs, sr, lr, lr_event, hr_event):
        sr_images, event_reconstruction_polarity = outputs
        b, f, c, h, w = hr_event.shape
        hr_event = hr_event.squeeze(2)
        common = event_reconstruction_polarity * hr_event
        common = -1.0 * common
        common = torch.relu(common)
        loss = torch.mean(common)
        return loss
