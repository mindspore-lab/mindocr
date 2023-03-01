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

from src.framework import ModuleBase
from src.utils import safe_load_yaml, get_shape_from_gear, get_device_id, check_valid_file, check_valid_dir


class CRNNInferProcess(ModuleBase):
    def __init__(self, config_path, msg_queue):
        super(CRNNInferProcess, self).__init__(config_path, msg_queue)
        self.model_list = defaultdict()
        self.static_method = True

    def init_self_args(self):
        base.mx_init()
        config = safe_load_yaml(self.config_path)
        crnn_config = config.get('crnn', {})
        if not crnn_config:
            raise ValueError(f'cannot find the crnn related config in config file')
        device_id = get_device_id(config, 'crnn')
        device_id = device_id if isinstance(device_id, int) else device_id[0]

        self.static_method = crnn_config.get('static_method', True)
        if self.static_method:
            model_path = crnn_config.get('model_path', '')
            model_dir = crnn_config.get('model_dir', '')
            if model_path and model_dir:
                raise ValueError('Both model_path and model_dir are found. Please use model_path to input the model '
                                 'fir path or model_dir to input the path to the folder containing only the '
                                 'CRNN/SVTR model files')
            if model_path and os.path.isfile(model_path):
                check_valid_file(model_path)
                model = base.model(model_path, device_id)
                batchsize, _, _ = get_shape_from_gear(model.model_gear())
                self.model_list[batchsize] = model

            if model_dir and os.path.isdir(model_dir):
                check_valid_dir(model_dir)
                for path in os.listdir(model_dir):
                    check_valid_file(os.path.join(model_dir, path))
                    model = base.model(os.path.join(model_dir, path), device_id)
                    batchsize, _, _ = get_shape_from_gear(model.model_gear())
                    self.model_list[batchsize] = model

        else:
            model_path = crnn_config.get('model_path', '')
            if model_path and os.path.isfile(model_path):
                check_valid_file(model_path)
                self.model_list[-1] = base.model(model_path, device_id)
            else:
                raise FileNotFoundError('CRNN model_path must be file when static method is False')
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
