#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
Description: decode module
Author: MindX SDK
Create: 2022
History: NA
"""

import numpy as np
from mindx.sdk import base, ImageProcessor

from deploy.infer_pipeline.framework.module_base import ModuleBase
from deploy.infer_pipeline.utils import safe_img_read, log, get_hw_of_img, safe_load_yaml, check_valid_file


class DecodeProcess(ModuleBase):
    def __init__(self, config_path, msg_queue):
        super().__init__(config_path, msg_queue)
        self.without_input_queue = False
        self.device = 'Ascend310P3'
        self.device_id = 0
        self.image_processor = None
        self.cost_time = 0

    def dvpp_decode(self, image_path):
        check_valid_file(image_path)
        dvpp_image_src = self.image_processor.decode(image_path, base.bgr)  # get the Image object from dvpp decoder
        dvpp_image_src.to_host()
        image_src = np.array(dvpp_image_src.to_tensor())
        image_src = image_src[0, :dvpp_image_src.original_height, :dvpp_image_src.original_width]
        return image_src

    def decode(self, image_path):
        if self.device == 'Ascend310P3':
            try:
                image_src = self.dvpp_decode(image_path)
            except RuntimeError:
                log.warning("dvpp not available, use opencv instead!")
                image_src = safe_img_read(image_path)
        else:
            image_src = safe_img_read(image_path)
        return image_src

    def init_self_args(self):
        base.mx_init()
        dic = safe_load_yaml(self.config_path)
        self.device = dic['device']
        if self.device == 'Ascend310P3':
            self.image_processor = ImageProcessor(self.device_id)
        elif self.device == 'Ascend310':
            pass
        else:
            log.warning("Unknown device type ,image decoding will be done using OpenCV.")

        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return
        image_path = input_data.image_path
        image_src = self.decode(image_path)
        h, w = get_hw_of_img(image_src)
        input_data.frame = image_src
        input_data.original_width = w
        input_data.original_height = h
        self.send_to_next_module(input_data)
