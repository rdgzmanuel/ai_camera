#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.num_classes = 80 # Number of classes in which YOLOX was trained (COCO dataset)

        # Testing/inference settings
        self.test_size = (640, 640)
        self.test_conf = 0.05 # Lower = more detections to filter later
        self.nmsthre = 0.45 
