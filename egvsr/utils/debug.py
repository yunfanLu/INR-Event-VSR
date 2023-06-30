#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project :EG-VSR 
@Author  :Yunfan Lu (yunfanlu@ust.hk)
@Date    :9/10/2022 10:11 PM 
"""

from pudb import set_trace
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_boolean("pudb", False, "Tag for debug.")


def pudb():
    if FLAGS.pudb:
        set_trace()
