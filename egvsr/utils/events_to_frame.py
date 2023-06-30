#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project :EG-VSR 
@Author  :Yunfan Lu (yunfanlu@ust.hk)
@Date    :9/12/2022 14:37 
"""
from typing import List, Tuple

import numpy as np


def event_stream_to_frames(
    events: List[np.ndarray],
    moments: int,
    resolution: Tuple[int, int],
    positive: int,
    negative: int,
):
    events = np.concatenate(events, axis=0)
    events = events[events[:, 0].argsort()]
    begin_time = events[0][0]
    end_time = events[-1][0]
    moment_time = (end_time - begin_time) / moments
    events_frames = []
    for i in range(moments):
        moment_begin = begin_time + i * moment_time
        moment_end = begin_time + (i + 1) * moment_time
        moment_events = events[events[:, 0] >= moment_begin]
        moment_events = moment_events[moment_events[:, 0] < moment_end]
        moment_count_frame = event_stream_to_a_count_frame(resolution, moment_events, positive, negative)
        events_frames.append(moment_count_frame)
    return events_frames


def event_stream_to_a_count_frame(
    resolution: Tuple[int, int],
    event: np.ndarray,
    positive: int,
    negative: int,
):
    H, W = resolution
    frame = np.zeros(shape=[H, W, 2])
    # Warning, Time may be Stack overflow.
    event = event.astype(int)
    positive_line = np.where(event[:, -1] == positive)
    negative_line = np.where(event[:, -1] == negative)
    positive_e = event[positive_line, 1:3].reshape(-1, 2)
    negative_e = event[negative_line, 1:3].reshape(-1, 2)

    positive_pos, positive_cnt = np.unique(positive_e, return_counts=True, axis=0)
    frame[positive_pos[:, 1], positive_pos[:, 0], 0] += positive_cnt[:]

    negative_pos, negative_cnt = np.unique(negative_e, return_counts=True, axis=0)
    frame[negative_pos[:, 1], negative_pos[:, 0], 1] += negative_cnt[:]
    # h, w, c -> c, h, w
    frame = frame.transpose(2, 0, 1).astype(np.float32)
    return frame
