#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
Description: Pre process class of the CRNN / SVTR (CTC decode base model)
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
from src.utils import get_batch_list_greedy, get_hw_of_img, safe_div, get_matched_gear_hw, padding_with_cv, normalize, \
    to_chw_image, expand, padding_batch, bgr_to_gray, safe_load_yaml, get_shape_from_gear, get_device_id, \
    check_valid_file, check_valid_dir, NORMALIZE_SCALE, NORMALIZE_MEAN, NORMALIZE_STD, log


class CRNNPreProcess(ModuleBase):
    def __init__(self, config_path, msg_queue):
        super(CRNNPreProcess, self).__init__(config_path, msg_queue)
        self.without_input_queue = False
        self.gear_list = []
        self.batchsize_list = []
        self.static_method = True
        self.model_height = 32
        self.model_channel = 3
        self.max_dot_gear = (0, 0)
        self.model_max_width = 2240
        self.model_min_width = 320
        self.scale = np.float32(NORMALIZE_SCALE)
        self.std = np.array(NORMALIZE_STD).astype(np.float32)
        self.mean = np.array(NORMALIZE_MEAN).astype(np.float32)

    def init_self_args(self):
        base.mx_init()
        config = safe_load_yaml(self.config_path)
        crnn_config = config.get('crnn', {})
        if not crnn_config:
            raise ValueError(f'cannot find the crnn related config in config file')
        device_id = get_device_id(config, 'crnn')
        device_id = device_id if isinstance(device_id, int) else device_id[self.instance_id % len(device_id)]

        self.static_method = crnn_config.get('static_method', True)
        if self.static_method:
            model_path = crnn_config.get('model_path', '')
            model_dir = crnn_config.get('model_dir', '')
            if model_path and model_dir:
                raise ValueError('Both model_path and model_dir are found. Please use model_path to input the model '
                                 'fir path or model_dir to input the path to the folder containing only the '
                                 'CRNN/SVTR model files')
            if os.path.isfile(model_path):
                check_valid_file(model_path)
                model = base.model(model_path, device_id)
                batchsize, channel, hw_list = get_shape_from_gear(model.model_gear())
                self.batchsize_list.append(batchsize)
                self.model_channel = channel
                self.gear_list = hw_list

            if os.path.isdir(model_dir):
                check_valid_dir(model_dir)
                for path in os.listdir(model_dir):
                    model = base.model(os.path.join(model_dir, path), device_id)
                    batchsize, channel, hw_list = get_shape_from_gear(model.model_gear())
                    self.batchsize_list.append(batchsize)
                    self.model_channel = channel
                    self.gear_list = hw_list
                    del model

            self.batchsize_list.sort()
            self.gear_list.sort()
            self.model_height = self.gear_list[0][0]
            self.max_dot_gear = self.gear_list[-1]
            self.model_max_width = self.max_dot_gear[1]
            self.model_min_width = self.gear_list[0][1]
        else:
            self.model_height = crnn_config.get('model_height', 32)
            model_min_width = crnn_config.get('model_min_width', 32)
            model_max_width = crnn_config.get('model_max_width', 4096)
            self.model_max_width = math.floor(safe_div(model_max_width, 32)) * 32
            self.model_min_width = math.ceil(safe_div(model_min_width, 32)) * 32
            if self.model_max_width < 1:
                log.error(f'model_max_width: {model_max_width} is less than 1, not valid')
                raise ValueError(f'model_max_width: {model_max_width} is less than 1, not valid')

            if self.model_min_width < 1:
                log.error(f'model_min_width: {model_min_width} is less than 1, not valid')
                raise ValueError(f'model_min_width: {model_min_width} is less than 1, not valid')

            if self.model_min_width > self.model_max_width:
                log.error('model_min_width must smaller than model_max_width')
                raise ValueError('model_min_width must smaller than model_max_width')

            if self.model_height < 1:
                log.error(f'model_height: {self.model_height} is less than 1, not valid')
                raise ValueError(f'model_height: {self.model_height} is less than 1, not valid')

        super().init_self_args()

    def get_max_width(self, image_list, max_wh_ratio):
        max_resize_w = 0
        max_width = max_wh_ratio * self.model_height
        for image in image_list:
            height, width = get_hw_of_img(image)
            ratio = safe_div(width, height)
            if math.ceil(ratio * self.model_height) > max_width:
                resize_w = max_width
            else:
                resize_w = math.ceil(self.model_height * ratio)
            max_resize_w = max(resize_w, max_resize_w)
            max_resize_w = max(min(max_resize_w, self.model_max_width), self.model_min_width)

        if self.static_method:
            _, gear_w = get_matched_gear_hw((self.model_height, max_resize_w), self.gear_list, self.max_dot_gear)
        else:
            gear_w = math.ceil(safe_div(max_resize_w, 32)) * 32
        return gear_w

    def preprocess(self, image_list, batchsize, max_resize_w, max_wh_ratio):
        input_list = []
        max_width = int(max_wh_ratio * self.model_height)
        for image in image_list:
            if self.model_channel == 1:
                image = bgr_to_gray(image)
            height, width = get_hw_of_img(image)
            ratio = safe_div(width, height)
            if math.ceil(ratio * self.model_height) > max_width:
                resize_w = max_width
            else:
                resize_w = math.ceil(self.model_height * ratio)
            resize_w = min(resize_w, self.model_max_width)
            crnn_image = cv2.resize(image, (resize_w, self.model_height))
            crnn_image = padding_with_cv(crnn_image, (self.model_height, max_resize_w))

            crnn_image = normalize(crnn_image, self.scale, self.std, self.mean)

            crnn_image = to_chw_image(crnn_image)

            input_list.append(crnn_image)

        input_array = expand(input_list)
        input_array = padding_batch(input_array, batchsize)
        return input_array

    def process(self, input_data):
        """
        split the sub image list to chunks by batch size and do the preprocess.
        If use dynamic model, the batch size will be the size of whole sub images list
        :param input_data: ProcessData
        :return: None
        """
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        sub_image_list = input_data.sub_image_list
        infer_res_list = input_data.infer_result
        max_wh_ratio = input_data.max_wh_ratio
        batch_list = [input_data.sub_image_size]
        if self.static_method:
            batch_list = get_batch_list_greedy(input_data.sub_image_size, self.batchsize_list)

        start_index = 0
        for batch in batch_list:
            upper_bound = min(start_index + batch, input_data.sub_image_size)
            split_input = sub_image_list[start_index:upper_bound]
            split_infer_res = infer_res_list[start_index:upper_bound]
            max_resize_w = self.get_max_width(split_input, max_wh_ratio)
            rec_model_inputs = self.preprocess(split_input, batch, max_resize_w, max_wh_ratio)

            send_data = ProcessData(sub_image_size=min(upper_bound - start_index, batch),
                                    image_path=input_data.image_path, image_total=input_data.image_total,
                                    infer_result=split_infer_res, input_array=rec_model_inputs,
                                    sub_image_total=input_data.sub_image_total, image_name=input_data.image_name,
                                    image_id=input_data.image_id)

            start_index += batch
            self.send_to_next_module(send_data)
