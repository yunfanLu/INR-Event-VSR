#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2022/10/19 16:06
import random

import torch
import torch.nn.functional as F
from absl.logging import info
from torch import nn

from egvsr.models.egsr.lcb import LightWeightCNNBackbone
from egvsr.models.egsr.ltb import MLABBlock4D
from egvsr.models.inr.random_scale_up_sampler import (
    RandomScaleUpSampler,
    get_WHT_coords,
)
from egvsr.models.utils import pair


class EventResidualConnection(nn.Module):
    def __init__(self, sample_number, out_channels, offset, layers):
        super(EventResidualConnection, self).__init__()
        self.sample_number = sample_number
        in_channels = (sample_number * 2 + 1) * 2
        self.offset = offset
        if layers == 2:
            self.sampler_module = nn.Sequential(
                nn.Conv2d(in_channels, 64, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(64, out_channels, 1, 1, 0),
            )
        elif layers == 3:
            self.sampler_module = nn.Sequential(
                nn.Conv2d(in_channels, 64, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(64, 64, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(64, out_channels, 1, 1, 0),
            )
        else:
            raise NotImplementedError

    def forward(self, events, key_timestamp, sr_h, sr_w):
        bz, fc, h, w = events.shape
        # (b, fc, h, w) -> (b, f, c, h, w)
        event_bfchw = events.reshape(bz, fc // 2, 2, h, w)
        # (b, f, c, h, w) -> (b, c, h, w, f)
        event_bchwt = event_bfchw.permute(0, 2, 3, 4, 1)

        dt = self.offset / self.sample_number
        sampled_events = []
        for i in range(2 * self.sample_number + 1):
            t = key_timestamp + (i - self.sample_number) * dt
            # Warning: the t may be out of range.
            t = min(1, max(t, -1))
            coord = get_WHT_coords(t, sr_h, sr_w)
            coord = coord.unsqueeze(0).repeat(bz, 1, 1, 1, 1).cuda()
            sampled_event = F.grid_sample(
                input=event_bchwt,
                grid=coord,
                align_corners=True,
                mode="bilinear",
            )
            sampled_events.append(sampled_event)
        sampled_events = torch.cat(sampled_events, dim=1)
        sampled_events = sampled_events.squeeze(2)
        sampled_event_feature = self.sampler_module(sampled_events)
        return sampled_event_feature


class RandomScaleSuperResolutionWithEvent(nn.Module):
    def __init__(
        self,
        in_frames,
        out_frames,
        is_include_bound,
        moments,
        event_channels,
        image_size,
        channels,
        n_feats,
        patch_size,
        is_shallow_fusion,
        interp_mode,
        random_up_sampler,
        sr_low_scale,
        sr_up_scale,
        time_bins,
        inr_channel,
        shallow_cnn_depth,
        shallow_transformer_layer,
        deep_cnn_depth,
        deep_transformer_layer,
        event_residual_connection,
        event_residual_sample_number,
        event_residual_offset,
        event_residual_layers,
        has_event_reconstruction,
    ):
        super(RandomScaleSuperResolutionWithEvent, self).__init__()

        # User Setting
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.out_frames_time = []
        t = 2.0 / (out_frames + 1)
        for i in range(out_frames):
            self.out_frames_time.append(-1 + (i + 1) * t)
        if is_include_bound:
            self.out_frames_time = [-1] + self.out_frames_time + [1]

        self.moments = moments
        self.time_bins = time_bins
        self.inr_channel = inr_channel

        self.patch_size = pair(patch_size)
        self.image_size = pair(image_size)

        self.n_feats = n_feats
        self.interp_mode = interp_mode
        self.random_up_sampler = random_up_sampler
        self.sr_low_scale = sr_low_scale
        self.sr_up_scale = sr_up_scale

        self.is_shallow_fusion = is_shallow_fusion

        self.shallow_cnn_depth = shallow_cnn_depth
        self.shallow_transformer_layer = shallow_transformer_layer
        self.deep_cnn_depth = deep_cnn_depth
        self.deep_transformer_layer = deep_transformer_layer

        self.event_residual_connection = event_residual_connection
        self.event_residual_sample_number = event_residual_sample_number
        self.event_residual_offset = event_residual_offset

        self.has_event_reconstruction = has_event_reconstruction
        # Infor the model arg
        self._info()

        # Network Part
        self.image_head = nn.Conv2d(channels * in_frames, n_feats, 3, padding=1)
        self.event_head = nn.Conv2d(moments * event_channels, n_feats, 3, padding=1)
        # shallow features
        self.shallow_image_cnn = LightWeightCNNBackbone(n_feats, depth=shallow_cnn_depth)
        self.shallow_event_cnn = LightWeightCNNBackbone(n_feats, depth=shallow_cnn_depth)
        if not is_shallow_fusion:
            self.shallow_fusion_transformer = None
        else:
            self.shallow_fusion_transformer = MLABBlock4D(
                self.image_size,
                self.patch_size,
                dim=patch_size * patch_size * n_feats * 2,
                depth=shallow_transformer_layer,
            )
        # Deep feature
        self.deep_image_cnn = LightWeightCNNBackbone(n_feats, depth=deep_cnn_depth)
        self.deep_event_cnn = LightWeightCNNBackbone(n_feats, depth=deep_cnn_depth)
        self.deep_fusion_transformer = MLABBlock4D(
            self.image_size,
            self.patch_size,
            dim=patch_size * patch_size * n_feats * 2,
            depth=deep_transformer_layer,
        )
        # To Latent Representation
        self.channel_up = nn.Conv2d(n_feats * 2, time_bins * inr_channel, 1)
        # random up
        self.up_model = RandomScaleUpSampler(
            decoder=self.random_up_sampler,
            interp_mode=self.interp_mode,
            has_event_feature=self.event_residual_connection,
            in_channel=inr_channel,
        )

        if event_residual_connection:
            self.event_residual_connection = EventResidualConnection(
                sample_number=event_residual_sample_number,
                out_channels=inr_channel,
                offset=event_residual_offset,
                layers=event_residual_layers,
            )

    def forward(self, image, event):
        if len(image.shape) == 5:
            b, f, c, h, w = image.shape
            image = image.reshape(b, f * c, h, w)
        if len(event.shape) == 5:
            b, f, c, h, w = event.shape
            event = event.reshape(b, f * c, h, w)
        # Head
        image_feature = self.image_head(image)
        event_feature = self.event_head(event)
        # Shallow
        image_shallow_feature = self.shallow_image_cnn(image_feature)
        event_shallow_feature = self.shallow_event_cnn(event_feature)

        if self.is_shallow_fusion:
            shallow_union_feature = torch.cat([image_shallow_feature, event_shallow_feature], dim=1)
            shallow_fusion_feature = self.shallow_fusion_transformer(shallow_union_feature)
            image_attention = shallow_fusion_feature[:, : self.n_feats, :, :]
            event_attention = shallow_fusion_feature[:, self.n_feats :, :, :]
            image_shallow_feature = image_shallow_feature + image_attention
            event_shallow_feature = event_shallow_feature + event_attention

        # Deep Feature
        image_deep_feature = self.deep_image_cnn(image_shallow_feature)
        event_deep_feature = self.deep_event_cnn(event_shallow_feature)
        deep_union_feature = torch.cat([image_deep_feature, event_deep_feature], dim=1)
        deep_fusion_feature = self.deep_fusion_transformer(deep_union_feature)
        # Latent feature
        latent_feature = self.channel_up(deep_fusion_feature)
        batch_size, ct, feature_h, feature_w = latent_feature.shape
        latent_feature = latent_feature.view(batch_size, self.time_bins, self.inr_channel, feature_h, feature_w)
        # (B, H, W, T, C)
        latent_feature = latent_feature.permute(0, 3, 4, 1, 2)

        random_up_scale = random.uniform(self.sr_low_scale, self.sr_up_scale)
        sr_h = int(self.image_size[0] * random_up_scale)
        sr_w = int(self.image_size[1] * random_up_scale)
        sr_image_list = []
        for t in self.out_frames_time:
            coord = get_WHT_coords(t=t, h=sr_h, w=sr_w)
            coord = coord.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).cuda()
            if self.event_residual_connection:
                event_feature = self.event_residual_connection(event, t, sr_h, sr_w)
            else:
                event_feature = None
            sr_image = self.up_model(latent_feature, coord, event_feature)
            sr_image_list.append(sr_image)
        sr_images = torch.cat(sr_image_list, dim=1)

        event_reconstruction_polarity = None
        if self.has_event_reconstruction:
            event_reconstruction_polarity = self.event_reconstruct(event, latent_feature)
        return sr_images, event_reconstruction_polarity

    def event_reconstruct(self, event, latent_feature):
        batch_size, fc, h, w = event.shape

        sr_gray_images = []
        dt = 2.0 / self.moments
        for i in range(self.moments + 1):
            t = -1.0 + i * dt
            sr_high_h = int(self.image_size[0] * self.sr_up_scale)
            sr_high_w = int(self.image_size[1] * self.sr_up_scale)
            coord = get_WHT_coords(t=t, h=sr_high_h, w=sr_high_w)
            coord = coord.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).cuda()
            if self.event_residual_connection:
                event_feature = self.event_residual_connection(event, t, sr_high_h, sr_high_w)
            else:
                event_feature = None
            sr_image = self.up_model(latent_feature, coord, event_feature)
            sr_gray_image = torch.mean(sr_image, dim=1, keepdim=True)
            sr_gray_images.append(sr_gray_image)

        sr_image_differ = []
        for i in range(self.moments):
            # (B, 1, H, W)
            sr_image_differ.append(sr_gray_images[i + 1] - sr_gray_images[i])
        # (B, N, H, W)
        sr_image_differ = torch.cat(sr_image_differ, dim=1)
        return sr_image_differ

    def _info(self):
        info(f"RandomScaleSuperResolutionWithEvent")
        info(f" in_frames                  : {self.in_frames}")
        info(f" out_frames                 : {self.out_frames}")
        info(f" out_frames_time            : {self.out_frames_time}")
        info(f" moments                    : {self.moments}")
        info(f" time_bins                  : {self.time_bins}")
        info(f" inr_channel                : {self.inr_channel}")
        info(f" patch_size                 : {self.patch_size}")
        info(f" image_size                 : {self.image_size}")
        info(f" n_feats                    : {self.n_feats}")
        info(f" sr_low_scale               : {self.sr_low_scale}")
        info(f" sr_up_scale                : {self.sr_up_scale}")
        info(f" random_up_sampler          : {self.random_up_sampler }")
        info(f" is_shallow_fusion          : {self.is_shallow_fusion}")
        info(f" shallow_cnn_depth          : {self.shallow_cnn_depth}")
        info(f" shallow_transformer_layer  : {self.shallow_transformer_layer}")
        info(f" deep_cnn_depth             : {self.deep_cnn_depth}")
        info(f" deep_transformer_layer     : {self.deep_transformer_layer}")
        info(f" event_residual_connection  : {self.event_residual_connection}")
        info(f"   residual_sample_number : {self.event_residual_sample_number}")
        info(f"   residual_offset        : {self.event_residual_offset}")
        info(f" has_event_reconstruction   : {self.has_event_reconstruction}")
