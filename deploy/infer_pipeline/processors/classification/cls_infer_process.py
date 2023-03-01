#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
Description: infer and post process class of the pp-ocr cls mobile 2.0 model
Author: MindX SDK
Create: 2022
History: NA
"""
import os

import cv2
import numpy as np
from mindx.sdk import base, Tensor

from deploy.infer_pipeline.framework import ModuleBase
from deploy.infer_pipeline.utils import safe_load_yaml, get_device_id, check_valid_file


class CLSInferProcess(ModuleBase):
    def __init__(self, config_path, msg_queue):
        super(CLSInferProcess, self).__init__(config_path, msg_queue)
        self.model = None
        self.static_method = True
        self.thresh = 0.9

    def init_self_args(self):
        base.mx_init()
        config = safe_load_yaml(self.config_path)
        cls_config = config.get('cls', {})
        if not cls_config:
            raise ValueError(f'cannot find the cls related config in config file')
        device_id = get_device_id(config, 'cls')
        device_id = device_id if isinstance(device_id, int) else device_id[self.instance_id % len(device_id)]
        model_path = cls_config.get('model_path', '')
        if model_path and os.path.isfile(model_path):
            check_valid_file(model_path)
            self.model = base.model(model_path, device_id)
        else:
            raise FileNotFoundError('cls model path must be a file')

        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        input_array = input_data.input_array
        inputs = [Tensor(input_array)]
        output = self.model.infer(inputs)
        output_array = output[0]
        output_array.to_host()
        output_array = np.array(output_array)

        for i in range(input_data.sub_image_size):
            if output_array[i, 1] > self.thresh:
                input_data.sub_image_list[i] = cv2.rotate(input_data.sub_image_list[i],
                                                          cv2.ROTATE_180)

        input_data.input_array = None
        # send the ready data to post module
        self.send_to_next_module(input_data)
