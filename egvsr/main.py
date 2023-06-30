#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project :EG-VSR
@Author  :Yunfan Lu (yunfanlu@ust.hk)
@Date    :9/11/2022 22:47
"""

import json
import os
import yaml
from absl import app
from absl import flags, logging
from absl.logging import info
from easydict import EasyDict
from pudb import set_trace

from egvsr.core.launch import ParallelLaunch

print("HELLO-HPC3")


FLAGS = flags.FLAGS

flags.DEFINE_string("yaml_file", None, "The config file.")
flags.DEFINE_string("RESUME_PATH", None, "The RESUME.PATH.")
flags.DEFINE_string("RESUME_TYPE", None, "The RESUME.PATH.")
flags.DEFINE_boolean("RESUME_SET_EPOCH", False, "The RESUME.PATH.")
flags.DEFINE_boolean("TEST_ONLY", False, "The test only.")
flags.DEFINE_boolean(f"VISUALIZE", False, "The visualization switch.")
#
flags.DEFINE_integer(f"TRAIN_BATCH_SIZE", None, "The train batch size.")
flags.DEFINE_integer(f"VAL_BATCH_SIZE", None, "The test batch size.")


def init_config(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    # 0. logging
    os.makedirs(FLAGS.log_dir, exist_ok=True)
    logging.set_verbosity(logging.DEBUG)
    info(f"log_dir: {FLAGS.log_dir}")
    logging.get_absl_handler().use_absl_log_file()
    config["SAVE_DIR"] = FLAGS.log_dir
    # 1. Resume
    if FLAGS.RESUME_PATH:
        config["RESUME"]["PATH"] = FLAGS.RESUME_PATH
        config["RESUME"]["TYPE"] = FLAGS.RESUME_TYPE
        config["RESUME"]["SET_EPOCH"] = FLAGS.RESUME_SET_EPOCH
    # 3. VISUALIZATION
    config["VISUALIZE"] = FLAGS.VISUALIZE
    # 4. Update batch size
    if FLAGS.TRAIN_BATCH_SIZE:
        info(f"Update TRAIN_BATCH_SIZE to {FLAGS.TRAIN_BATCH_SIZE}")
        config["TRAIN_BATCH_SIZE"] = FLAGS.TRAIN_BATCH_SIZE
    if FLAGS.VAL_BATCH_SIZE:
        info(f"Update VAL_BATCH_SIZE to {FLAGS.VAL_BATCH_SIZE}")
        config["VAL_BATCH_SIZE"] = FLAGS.VAL_BATCH_SIZE
    # 5. TEST_ONLY
    if FLAGS.TEST_ONLY:
        config["TEST_ONLY"] = FLAGS.TEST_ONLY

    info(f"Launch Config: {json.dumps(config, indent=4, sort_keys=True)}")
    return EasyDict(config)


def main(args):
    # set_trace()
    config = init_config(FLAGS.yaml_file)
    # 0. logging
    # 1. init launcher
    launcher = ParallelLaunch(config)
    launcher.run()


if __name__ == "__main__":
    app.run(main)
