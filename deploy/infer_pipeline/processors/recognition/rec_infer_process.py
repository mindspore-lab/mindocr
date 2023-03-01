#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
Description: infer process class of the CRNN / SVTR (CTC decode base model)
Author: MindX SDK
Create: 2022
History: NA
"""
import os
from collections import defaultdict

import numpy as np
from mindx.sdk import base, Tensor

from deploy.infer_pipeline.framework import ModuleBase
from deploy.infer_pipeline.utils import safe_load_yaml, get_shape_info, get_device_id, check_valid_file, check_valid_dir


class RecInferProcess(ModuleBase):
    def __init__(self, config_path, msg_queue):
        super(RecInferProcess, self).__init__(config_path, msg_queue)
        self.model_list = defaultdict()
        self.static_method = True

    def get_single_model(self, filename, device_id):
        check_valid_file(filename)
        model = base.model(filename, device_id)
        desc, shape_info = get_shape_info(model.input_shape(0), model.model_gear())

        if desc == "dynamic_shape":
            self.static_method = False
            self.model_list[-1] = model
        else:
            self.static_method = True
            self.model_list[shape_info[0]] = model

        return model

    def init_self_args(self):
        base.mx_init()
        config = safe_load_yaml(self.config_path)
        crnn_config = config.get('crnn', {})
        if not crnn_config:
            raise ValueError(f'cannot find the crnn related config in config file')
        device_id = get_device_id(config, 'crnn')
        device_id = device_id if isinstance(device_id, int) else device_id[0]

        model_path = crnn_config.get('model_dir', '') or crnn_config.get('model_path', '')

        if os.path.isfile(model_path):
            self.get_single_model(model_path, device_id)

        if os.path.isdir(model_path):
            check_valid_dir(model_path)
            for path in os.listdir(model_path):
                filename = os.path.join(model_path, path)
                check_valid_file(filename)
                self.get_single_model(filename, device_id)

        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        input_array = input_data.input_array
        batchsize, _, _, _ = input_array.shape
        inputs = [Tensor(input_array)]
        if self.static_method:
            if batchsize not in self.model_list:
                return
            output = self.model_list[batchsize].infer(inputs)
        else:
            output = self.model_list[-1].infer(inputs)
        output_array = output[0]
        output_array.to_host()
        output_array = np.array(output_array)
        # send the ready data to post module
        input_data.output_array = output_array
        input_data.input_array = None
        self.send_to_next_module(input_data)
