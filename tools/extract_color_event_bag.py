#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Project :EG-VSR
# Author  :Yunfan Lu (yunfanlu@ust.hk)
# Date    :9/11/2022 17:29

from os import listdir, makedirs
from os.path import join, isfile

import cv2
import numpy as np
import rosbag
from absl import app
from absl.logging import info, error

from config import global_path as gp


def color_image_msg_to_png(msg, image_path):
    image_buffer = np.frombuffer(msg.data, np.uint8)
    hight, width, channel = msg.height, msg.width, 3
    image = np.reshape(image_buffer, [hight, width, channel])
    cv2.imwrite(image_path, image)


def events_msg_to_event_npy(msg, event_path):
    events = []
    for i, event in enumerate(msg.events):
        item = [
            float(event.ts.to_nsec()),
            event.x,
            event.y,
            1 if event.polarity else 0,
        ]
        events.append(item)
    # dtype = [('t', float), ('x', int), ('y', int), ('p', int)]
    events = np.array(events, dtype=np.float)
    np.save(event_path, events)


def extract_color_image_and_event(bag_file, extract_folder):
    info(f"mkdir {extract_folder}")
    makedirs(extract_folder, exist_ok=True)
    time_stamp = 0
    for topic, msg, t in rosbag.Bag(bag_file).read_messages():
        if topic == "/dvs/image_color":
            image_path = join(extract_folder, f"{t.to_nsec()}.png")
            if not isfile(image_path):
                info(f"Color Image: {t}. {image_path}")
                color_image_msg_to_png(msg, image_path)
        elif topic == "/dvs/events":
            event_path = join(extract_folder, f"{t.to_nsec()}.npy")
            if not isfile(event_path):
                info(f"Events     : {t}. {event_path}")
                events_msg_to_event_npy(msg, event_path)
        else:
            continue
        if t.to_nsec() < time_stamp:
            error(f"{t.to_nsec()} > {time_stamp}")
            error(f"  {t.to_nsec() - time_stamp}")
        time_stamp = t.to_nsec()


def main(args):
    color_events_bag_folder = gp.color_event_bags
    event_color_image_folder = gp.color_event_dataset
    bags = listdir(color_events_bag_folder)
    info(f"Bag count: {len(bags)}")
    for ibag, bag in enumerate(bags):
        bag_name = bag.replace(".bag", "")
        info(f"[{ibag}], bag:{bag}, ban_name:{bag_name}")
        bag_file = join(color_events_bag_folder, bag)
        extract_folder = join(event_color_image_folder, bag_name)
        info(f"Extraction:" f"  {bag_file} ->" f"  {extract_folder}")
        extract_color_image_and_event(bag_file, extract_folder)


if __name__ == "__main__":
    app.run(main)
