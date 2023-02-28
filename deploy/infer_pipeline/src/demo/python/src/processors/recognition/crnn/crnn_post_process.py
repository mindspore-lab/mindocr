#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
Description: Post process class of the CRNN / SVTR (CTC decode base model)
Author: MindX SDK
Create: 2022
History: NA
"""
import os.path

import numpy as np
from src.utils import array_to_texts, safe_load_yaml, file_base_check, log

from src.framework import ModuleBase


class CRNNPostProcess(ModuleBase):
    def __init__(self, config_path, msg_queue):
        super(CRNNPostProcess, self).__init__(config_path, msg_queue)
        self.without_input_queue = False
        self.labels = [' ']

    def init_self_args(self):
        config = safe_load_yaml(self.config_path)
        crnn_config = config.get('crnn', {})
        if not crnn_config:
            raise ValueError(f'cannot find the crnn related config in config file')
        label_path = crnn_config.get('dict_path', '')
        if label_path and os.path.isfile(label_path):
            file_base_check(label_path)
        else:
            raise FileNotFoundError('CRNN dict_path must be a file')
        with open(label_path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip("\n").strip("\r\n")
                self.labels.append(line)
        self.labels.append(' ')
        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        output_array = input_data.output_array
        if len(output_array.shape) == 3:
            output_array = np.argmax(output_array, axis=2, keepdims=False)
            log.warn(
                f'Running argmax operator in cpu. Please use the insert_argmax script to add the argmax operator '
                f'into the model to improve the inference performance.')

        rec_result = array_to_texts(output_array, self.labels, input_data.sub_image_size)
        for coord, text in zip(input_data.infer_result, rec_result):
            coord.append(text)

        self.send_to_next_module(input_data)
