#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
Description: pre process class of the pp-ocr cls mobile 2.0 model
Author: MindX SDK
Create: 2022
History: NA
"""

import math
import os

import cv2
import numpy as np
from mindx.sdk import base

from src.data_type.process_data import ProcessData
from src.framework import ModuleBase
from src.utils import get_batch_list_greedy, get_hw_of_img, safe_div, padding_with_cv, normalize, \
    to_chw_image, expand, padding_batch, bgr_to_gray, safe_load_yaml, get_batch_from_gear, get_device_id, \
    check_valid_file, NORMALIZE_MEAN, NORMALIZE_STD, NORMALIZE_SCALE


class CLSPreProcess(ModuleBase):
    def __init__(self, config_path, msg_queue):
        super(CLSPreProcess, self).__init__(config_path, msg_queue)
        self.without_input_queue = False
        self.gear_list = []
        self.batchsize_list = []
        self.model_height = 48
        self.model_channel = 3
        self.model_width = 192
        self.scale = np.float32(NORMALIZE_SCALE)
        self.std = np.array(NORMALIZE_STD).astype(np.float32)
        self.mean = np.array(NORMALIZE_MEAN).astype(np.float32)

    def init_self_args(self):
        base.mx_init()
        config = safe_load_yaml(self.config_path)
        cls_config = config.get('cls', {})
        if not cls_config:
            raise ValueError(f'cannot find the cls related config in config file')
        device_id = get_device_id(config, 'cls')
        device_id = device_id if isinstance(device_id, int) else device_id[0]
        model_path = cls_config.get('model_path', '')
        if model_path and os.path.isfile(model_path):
            check_valid_file(model_path)
            model = base.model(model_path, device_id)
        else:
            raise FileNotFoundError('cls model path must be a file')
        batchsize_list, channel, height, width = get_batch_from_gear(model.model_gear())
        self.batchsize_list = batchsize_list
        self.model_channel = channel
        self.model_height = height
        self.model_width = width
        del model
        self.batchsize_list.sort()
        super().init_self_args()

    def preprocess(self, image_list, batchsize):
        input_list = []
        for image in image_list:
            if self.model_channel == 1:
                image = bgr_to_gray(image)
            h, w = get_hw_of_img(image)
            ratio = safe_div(w, h)
            if math.ceil(ratio * self.model_height) > self.model_width:
                resize_w = self.model_width
            else:
                resize_w = math.ceil(self.model_height * ratio)

            cls_image = cv2.resize(image, (resize_w, self.model_height))

            cls_image = padding_with_cv(cls_image, (self.model_height, self.model_width))

            cls_image = normalize(cls_image, self.scale, self.std, self.mean)

            cls_image = to_chw_image(cls_image)

            input_list.append(cls_image)

        input_array = expand(input_list)
        input_array = padding_batch(input_array, batchsize)
        return input_array

    def process(self, input_data):
        """
        split the sub image list to chunks by batch size and do the preprocess.
        :param input_data: ProcessData
        :return: None
        """
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        sub_image_list = input_data.sub_image_list
        infer_res_list = input_data.infer_result
        batch_list = get_batch_list_greedy(input_data.sub_image_size, self.batchsize_list)
        start_index = 0
        for batch in batch_list:
            upper_bound = min(start_index + batch, input_data.sub_image_size)
            split_input = sub_image_list[start_index:upper_bound]
            split_infer_res = infer_res_list[start_index:upper_bound]
            cls_model_inputs = self.preprocess(split_input, batch)
            send_data = ProcessData(sub_image_size=len(split_input), sub_image_list=split_input,
                                    image_path=input_data.image_path, image_total=input_data.image_total,
                                    infer_result=split_infer_res, input_array=cls_model_inputs,
                                    sub_image_total=input_data.sub_image_total, image_name=input_data.image_name,
                                    image_id=input_data.image_id, max_wh_ratio=input_data.max_wh_ratio)

            start_index += batch
            self.send_to_next_module(send_data)
