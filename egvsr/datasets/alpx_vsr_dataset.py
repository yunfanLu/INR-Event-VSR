#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2022/11/2 11:12
import logging
from os import listdir
from os.path import join, isdir

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from absl.logging import info
from torch.utils.data import Dataset

logging.getLogger("PIL").setLevel(logging.WARNING)


def get_alpx_vsr_dataset(
    alpx_vsr_root,
    moments,
    in_frame,
    future_frame,
    past_frame,
    scale,
    random_crop_resolution,
    high_resolution,
    low_resolution,
    evaluation_visualization,
):
    all_videos = sorted(listdir(alpx_vsr_root))
    # 6 test video and 20 train video
    test_videos = [
        "20221022152648846",
        "20221022153331562",
        "20221022160838823",
        "20221022160418501",
        "20221022161700116",
        "20221022160714153",
    ]
    train_videos = []
    for v in all_videos:
        if v in test_videos:
            continue
        if not isdir(join(alpx_vsr_root, v)):
            continue
        train_videos.append(v)

    info(f"train_videos ({len(train_videos)}): {train_videos}")
    info(f"test_videos  ({len(test_videos)}): {test_videos}")

    train_dataset = AlpxVideoSRDataset(
        alpx_vsr_root,
        train_videos,
        moments,
        in_frame,
        future_frame,
        past_frame,
        scale,
        random_crop_resolution,
        high_resolution,
        low_resolution,
        evaluation_visualization,
    )
    test_dataset = AlpxVideoSRDataset(
        alpx_vsr_root,
        test_videos,
        moments,
        in_frame,
        future_frame,
        past_frame,
        scale,
        random_crop_resolution,
        high_resolution,
        low_resolution,
        evaluation_visualization,
    )
    return train_dataset, test_dataset


class AlpxVideoSRDataset(Dataset):
    def __init__(
        self,
        alpx_vsr_root,
        videos,
        moments,
        in_frame,
        future_frame,
        past_frame,
        scale,
        random_crop_resolution,
        high_resolution,
        low_resolution,
        evaluation_visualization,
    ):
        super(AlpxVideoSRDataset, self).__init__()
        self.moments = moments
        self.alpx_vsr_root = alpx_vsr_root
        self.videos = videos
        self.in_frame = in_frame
        self.future_frame = future_frame
        self.past_frame = past_frame
        self.random_crop_resolution = random_crop_resolution
        self.high_resolution = high_resolution
        self.low_resolution = low_resolution
        self.scale = scale
        assert self.random_crop_resolution[0] >= self.high_resolution[0]
        assert self.random_crop_resolution[1] >= self.high_resolution[1]
        assert self.low_resolution[0] * self.scale == self.high_resolution[0]
        assert self.low_resolution[1] * self.scale == self.high_resolution[1]
        # static values
        self.event_resolution = (1224, 1632)
        self.positive = 2
        self.negative = 1
        self.evaluation_visualization = evaluation_visualization

        self.items = self._generate_items()
        self._info()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_paths, event_paths = self.items[index]
        h, w = self.event_resolution
        images = []
        for path in image_paths:
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"image is None: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (w, h))
            images.append(image)

        events = []
        for path in event_paths:
            event = np.load(path)
            event[event == self.negative] = -1
            event[event == self.positive] = 1
            event = np.fliplr(cv2.rotate(event, cv2.ROTATE_90_CLOCKWISE))
            events.append(event)
        if len(events) < self.moments:
            for i in range(self.moments - len(events)):
                events.append(np.zeros_like(events[0]))

        lr, lr_events, hr, hr_events = self._random_crop(images, events)
        return lr, lr_events, hr, hr_events

    def _random_crop(self, images, events):
        h, w = self.event_resolution
        crop_h, crop_w = self.random_crop_resolution
        hr_h, hr_w = self.high_resolution
        lr_h, lr_w = hr_h // self.scale, hr_w // self.scale

        if self.evaluation_visualization:
            x, y = 0, 0
        else:
            x = np.random.randint(0, h - crop_h - 1)
            y = np.random.randint(0, w - crop_w - 1)
        # info(f" h: {h}, w: {w}, crop_h:{crop_h}, crop_w:{crop_w}")
        # info(f" x:{x}, y:{y}, -> {x + crop_h}, {y + crop_w}")
        # info(f" lr_h:{lr_h}, lr_w:{lr_w}")
        # info(f" image[0]:{images[0].shape}")
        # info(f" event[0]:{events[0].shape}")
        # crop
        crop_images = []
        for i in range(len(images)):
            image = torch.from_numpy(images[i]).float()
            image = image.permute(2, 0, 1)
            crop_image = image[:, x : x + crop_h, y : y + crop_w]
            crop_images.append(crop_image / 255.0)

        crop_events = []
        for i in range(len(events)):
            event = torch.from_numpy(np.ascontiguousarray(events[i])).float()
            event = event.unsqueeze(0)
            crop_event = event[:, x : x + crop_h, y : y + crop_w]
            crop_events.append(crop_event)
        # resize
        crop_images = torch.stack(crop_images, dim=0)
        hr_images = F.interpolate(
            crop_images[self.past_frame : -self.future_frame],
            size=(hr_h, hr_w),
            mode="bilinear",
            align_corners=False,
        )
        lr_images = F.interpolate(
            crop_images,
            size=(lr_h, lr_w),
            mode="bilinear",
            align_corners=False,
        )

        crop_events = torch.stack(crop_events, dim=0)
        lr_events = F.interpolate(
            crop_events,
            size=(lr_h, lr_w),
            mode="bilinear",
            align_corners=False,
        )
        hr_events = F.interpolate(
            crop_events,
            size=(hr_h, hr_w),
            mode="bilinear",
            align_corners=False,
        )
        return lr_images, lr_events, hr_images, hr_events

    def _generate_items(self):
        samples = []
        for video in self.videos:
            video_path = join(self.alpx_vsr_root, video)
            samples += self._generate_video_items(video_path)
        return samples

    def _generate_video_items(self, video_path):
        items = []
        files = sorted(listdir(video_path))
        for i in range(len(files)):
            if not files[i].endswith("frame.png"):
                continue
            frame_indexes = []
            for j in range(i + 1, len(files)):
                if not files[j].endswith("frame.png"):
                    continue
                frame_indexes.append(j)
                if len(frame_indexes) == self.in_frame:
                    break
            # 1. Check has enough frames.
            if len(frame_indexes) != self.in_frame:
                break
            # 2. Check the event in neighbor frames is average.
            is_average = True
            for k in range(self.in_frame - 1):
                index_step = frame_indexes[k + 1] - frame_indexes[k]
                # 20, 50, -> 31, 34
                if index_step <= 20 or index_step >= 50:
                    is_average = False
                    break
            if not is_average:
                continue
            # 3. Generate items.
            item = [[], []]
            for k in range(min(frame_indexes), max(frame_indexes) + 1):
                frame_name = files[k]
                if frame_name.endswith("frame.png"):
                    item[0].append(join(video_path, frame_name))
                elif frame_name.endswith("events.npy"):
                    item[1].append(join(video_path, frame_name))
            items.append(item)
        return items

    def _info(self):
        info(f"Init AlpxVideoSRDataset.")
        info(f"  alpx_vsr_root: {self.alpx_vsr_root}")
        info(f"  videos: {self.videos}")
        info(f"  in_frame: {self.in_frame}")
        info(f"  future_frame: {self.future_frame}")
        info(f"  past_frame: {self.past_frame}")
        info(f"  scale: {self.scale}")
        info(f"  random_crop_resolution: {self.random_crop_resolution}")
        info(f"  low_resolution: {self.low_resolution}")
        info(f"  Length: {len(self.items)}")
