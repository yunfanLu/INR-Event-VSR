import logging
from os import listdir
from os.path import join
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from absl.logging import info
from torch.utils.data import Dataset

from torchvision.transforms.functional import to_tensor

from egvsr.utils.events_to_frame import event_stream_to_frames

logging.getLogger("PIL").setLevel(logging.WARNING)


def get_ced_time_stamp(ced_file):
    file_name = Path(ced_file).stem
    return float(file_name)


def get_ced_dataset(ced_root, in_frame, future_frame, past_frame, scale, moments, is_mini):
    train_video, test_videos = [], []
    all_video = sorted(listdir(ced_root))

    # These two video have different resolution with other videos.
    all_video.remove("driving_city_3")
    all_video.remove("calib_fluorescent_dynamic")

    test_videos = [
        "people_dynamic_wave",
        "indoors_foosball_2",
        "simple_wires_2",
        "people_dynamic_dancing",
        "people_dynamic_jumping",
        "simple_fruit_fast",
        "outdoor_jumping_infrared_2",
        "simple_carpet_fast",
        "people_dynamic_armroll",
        "indoors_kitchen_2",
        "people_dynamic_sitting",
    ]

    for i, video in enumerate(all_video):
        if video in test_videos:
            continue
        if i % 8 == 0:
            test_videos.append(video)
        else:
            train_video.append(video)

    train_dataset = ColorEventSRDataset(
        ced_root,
        train_video,
        in_frame,
        future_frame,
        past_frame,
        scale,
        moments,
        is_train=True,
        is_mini=is_mini,
    )
    test_dataset = ColorEventSRDataset(
        ced_root,
        test_videos,
        in_frame,
        future_frame,
        past_frame,
        scale,
        moments,
        is_train=False,
        is_mini=is_mini,
    )
    return train_dataset, test_dataset


class ColorEventSRDataset(Dataset):
    @property
    def height(self):
        return 260

    @property
    def width(self):
        return 346

    def __init__(
        self,
        ced_root,
        videos,
        in_frame,
        future_frame,
        past_frame,
        scale,
        moments,
        is_train,
        is_mini,
    ):
        """
        The CED dataset resolution is 346x260
        :param ced_root:
        :param in_frame:
        :param future_frame:
        :param past_frame:
        :param scale:
        :param moments:
        """
        super(ColorEventSRDataset, self).__init__()
        assert in_frame >= 1 and in_frame % 2 == 1, f"in_frame({in_frame}) must be a positive and odd paper."
        assert future_frame + past_frame < in_frame, (
            f"future_frame({future_frame}) and past_frame({past_frame})" f"must be less than in_frame({in_frame})."
        )

        self.ced_root = ced_root
        self.videos = videos
        self.is_train = is_train
        # Image config
        self.in_frame = in_frame
        self.future_frame = future_frame
        self.past_frame = past_frame
        self.scale = scale
        self.high_resolution = (260, 346)
        self.low_resolution = (260 // scale, 346 // scale)
        # Event config
        self.moments = moments
        self.is_mini = is_mini
        # Generate the inference and training items.
        self.items = self._generate_items()
        if self.is_mini == 1:
            pass
        elif is_mini == 5:
            self.items = self.items[::5]  # Only use 1/5 data for mini dataset.
        elif is_mini == 20:
            self.items = self.items[::20]
        else:
            raise ValueError(f"Unknown is_mini: {is_mini}")

        self.positive = 1
        self.negative = 0

        info(f"ColorEventSRDataset:")
        info(f"  - ced_root: {self.ced_root}")
        info(f"  - number of videos: {len(self.videos)}")
        info(f"  - is train: {self.is_train}")
        info(f"  - in_frame: {self.in_frame}")
        info(f"  - scale: {self.scale}")
        info(f"     - up: {self.low_resolution}->{self.high_resolution}")
        info(f"  - moments: {self.moments}")
        info(f"  - event: single polarity")
        info(f"     - positive: {self.positive}")
        info(f"     - negative: {self.negative}")
        info(f"  - items: {len(self.items)}")

    def __getitem__(self, index):
        image_paths, events = self.items[index]
        # info(f"images[{index}]:")
        # for image in images:
        #     info(f"  - {image}")
        # info(f"events[{index}]:")
        # for event in events:
        #     info(f"  - {event}")

        # Image loading
        images = [Image.open(image) for image in image_paths]

        # Bad data
        # for i in range(len(images)):
        #     if images[i].size != (self.width, self.height):
        #         warning(f"Image size is not correct: {images[i].size}")
        #         warning(f"  - image: {image_paths[i]}")
        #         return self[(index + 1) % len(self)]

        # Attention, the input of resize function is (width, height)!
        (low_height, low_width) = self.low_resolution
        lr = [img.resize((low_width, low_height)) for img in images]
        hr = images[self.past_frame : -self.future_frame]
        lr = [to_tensor(img) for img in lr]
        hr = [to_tensor(img) for img in hr]
        lr = torch.stack(lr, dim=0)
        hr = torch.stack(hr, dim=0)
        # Event loading
        events = [np.load(event) for event in events]
        events = event_stream_to_frames(
            events,
            self.moments,
            self.high_resolution,
            self.positive,
            self.negative,
        )
        events = np.stack(events, axis=0)
        hr_events = torch.from_numpy(events)
        # events = events[:, :, ::2, ::2]
        lr_events = F.interpolate(
            hr_events,
            size=self.low_resolution,
            mode="bilinear",
            align_corners=False,
        )

        return lr, lr_events, hr, hr_events

    def __len__(self):
        return len(self.items)

    def _generate_items(self) -> List:
        items = []
        for video_name in self.videos:
            video_folder = join(self.ced_root, video_name)
            video_items = self._generate_from_video(video_folder)
            if len(video_items):
                items.extend(video_items)
        return items

    def _generate_from_video(self, video_folder):
        files = sorted(listdir(video_folder))
        files = [f for f in files if (f.endswith(".png") or f.endswith(".npy"))]
        items = []
        length = len(files)
        for i in range(1, length):
            left = i
            right = -100
            if files[i].endswith(".png"):
                count = 1
                for j in range(i + 1, length):
                    if files[j].endswith(".png"):
                        count += 1
                    else:
                        continue
                    if count == self.in_frame:
                        right = j
                        break
            else:
                continue
            # Generate training item
            # e, i, i, i, e,
            # e, e, i, e, i
            # e, i, e, i, e, i, e
            if length - 2 > right > left > 0 and files[left - 1].endswith(".npy") and files[right + 1].endswith("npy"):
                item = [[], []]
                for k in range(left - 1, right + 2):
                    if files[k].endswith("png"):
                        item[0].append(join(video_folder, files[k]))
                    elif files[k].endswith("npy"):
                        item[1].append(join(video_folder, files[k]))
                    else:
                        pass
                items.append(item)
        return items
