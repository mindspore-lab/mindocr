#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
Description: infer process class of DB Net
Author: MindX SDK
Create: 2022
History: NA
"""

import numpy as np
from mindx.sdk import base, Tensor

from src.framework import ModuleBase
from src.utils import safe_load_yaml, get_matched_gear_hw, get_shape_from_gear, padding_with_np, get_device_id


class DBNetInferProcess(ModuleBase):
    def __init__(self, config_path, msg_queue):
        super(DBNetInferProcess, self).__init__(config_path, msg_queue)
        self.without_input_queue = False
        self.model = None
        self.gear_list = None
        self.model_channel = None
        self.max_dot_gear = None

    def init_self_args(self):
        base.mx_init()
        config = safe_load_yaml(self.config_path)
        dbnet_config = config.get('dbnet', {})
        if not dbnet_config:
            raise ValueError(f'cannot find the dbnet related config in config file')
        model_path = dbnet_config.get('model_path', '')

        device_id = get_device_id(config, 'dbnet')
        device_id = [device_id] if isinstance(device_id, int) else device_id
        self.model = base.model(model_path, device_id[self.instance_id % len(device_id)])
        batchsize, channel, hw_list = get_shape_from_gear(self.model.model_gear())
        self.gear_list = hw_list
        self.model_channel = channel
        self.max_dot_gear = max([(h, w) for h, w in hw_list], key=lambda x: x[0] * x[1])

        self.warmup()
        super().init_self_args()

    def warmup(self):
        dummy_tensor = np.random.randn(1, self.model_channel, self.max_dot_gear[0], self.max_dot_gear[1]).astype(
            np.float32)
        inputs = [Tensor(dummy_tensor)]
        self.model.infer(inputs)

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return
        input_array = input_data.input_array
        n, c, h, w = input_array.shape

        matched_gear = get_matched_gear_hw((h, w), self.gear_list, self.max_dot_gear)
        input_array = padding_with_np(input_array, matched_gear)
        inputs = [Tensor(input_array)]
        output = self.model.infer(inputs)
        if not output:
            output = self.model.infer(inputs)
        output_array = output[0]
        output_array.to_host()
        output_array = np.array(output_array)

        output_array = output_array[:, :, :input_data.resize_h, :input_data.resize_w]

        # send the ready data to post module
        input_data.output_array = output_array
        input_data.input_array = None
        self.send_to_next_module(input_data)
