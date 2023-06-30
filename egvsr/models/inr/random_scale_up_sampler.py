from turtle import forward
import torch
import torch.nn.functional as F
from torch import nn

from egvsr.models.egsr.scale import Weight
from egvsr.models.inr.decoders import CNNDecoder, MLPDecoder, SirenDecoder


def get_WHT_coords(t: float, h: int, w: int):
    """
    :param t: The timestamp of the sampled frame.
    :param h: The height of the sampled frame, which is not the original height.
    :param w: The width of the sampled frame, which is not the original width.
    :return: a coords tensor of shape (1, h, w, 3), where the last dimension
             is (t, w, h).
    """
    assert t >= -1 and t <= 1, f"Time t should be in [-1, 1], but got {t}."
    # (1, h, w, 3):
    #   1 means one time stamps.
    #   h and w means the height and width of the image.
    #   3 means the w, h, and t coordinates. The order is important.
    grid_map = torch.zeros(1, h, w, 3) + t

    h_coords = torch.linspace(-1, 1, h)
    w_coords = torch.linspace(-1, 1, w)
    mesh_h, mesh_w = torch.meshgrid([h_coords, w_coords])
    # The feature is H W T, so the coords order is (t, w, h)
    # grid_map \in R^{1, h, w, 3}, grid_map[:, :, :, i] is (t, w, h)
    grid_map[:, :, :, 1:] = torch.stack((mesh_w, mesh_h), 2)
    return grid_map.float()


class RandomScaleUpSampler(nn.Module):
    def __init__(
        self,
        decoder="identity",
        interp_mode="bilinear",
        has_event_feature=False,
        in_channel=64,
        hidden_layers=3,
    ):
        super().__init__()
        assert interp_mode == "nearest" or interp_mode == "bilinear"

        if interp_mode == "nearest":
            in_channel += 3

        if decoder == "mlp":
            self.decoder = MLPDecoder(
                in_features=in_channel,
                hidden_features=in_channel,
                hidden_layers=hidden_layers,
                out_features=3,
            )
        elif decoder == "siren":
            self.decoder = SirenDecoder(
                in_features=in_channel,
                hidden_features=in_channel,
                hidden_layers=hidden_layers,
                out_features=3,
            )
        elif decoder == "cnn":
            self.decoder = CNNDecoder(
                in_features=in_channel,
                hidden_features=in_channel,
                hidden_layers=hidden_layers,
                out_features=3,
            )
        else:
            self.decoder = nn.Identity()

        self.interp_mode = interp_mode
        self.has_event_feature = has_event_feature
        # if we use the event skip connection, we should use the wight
        if has_event_feature:
            self.inr_weight = Weight(1.0)
            self.event_weight = Weight(0.1)

    def forward(self, feature, coords, event_feature=None):
        """
        Args:
            feature: (B, H, W, T, C)
            coords: (B, W, H, T, 3) \in {t, h, w}
        Returns:
            out: (B, C, N, H, W)
        """
        B, H, W, T, C = feature.shape
        # (B, H, W, T, C) -> (B, C, H, W, T)
        feature = feature.permute(0, 4, 1, 2, 3)

        if self.interp_mode == "bilinear":
            # (B, C, H, W, T) -> (B, C, 1, H, W)
            sampled_feature = F.grid_sample(input=feature, grid=coords, align_corners=True, mode="bilinear")
        elif self.interp_mode == "nearest":
            # as in LIIF
            # (B, C, 1, H, W)
            sampled_feature = F.grid_sample(input=feature, grid=coords, mode="nearest", align_corners=True)
            coords = coords.permute(0, 4, 1, 2, 3)
            sampled_feature = torch.cat([sampled_feature, coords], dim=1)
        else:
            raise NotImplementedError

        # (B, C, 1, H, W) -> (B, H, W, 1, C)
        sampled_feature = sampled_feature.permute(0, 3, 4, 2, 1)
        # (B, H, W, 1, C) -> (B, H, W, C) -> (B, C, H, W)
        sampled_feature = sampled_feature.squeeze(3).permute(0, 3, 1, 2)

        if self.has_event_feature and (event_feature is not None):
            sampled_feature_w = self.inr_weight(sampled_feature)
            event_feature_w = self.event_weight(event_feature)
            sampled_feature = sampled_feature_w + event_feature_w

        sampled_feature = self.decoder(sampled_feature)
        return sampled_feature
