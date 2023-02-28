#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
Description: collect the mini batch, save the infer result and send the stop message to the manager module.
Author: MindX SDK
Create: 2022
History: NA
"""

import os
from collections import defaultdict
from ctypes import c_uint64
from multiprocessing import Manager

from src.data_type import StopData, ProcessData, ProfilingData
from src.framework.module_base import ModuleBase
from src.utils import safe_list_writer, log


class CollectProcess(ModuleBase):
    def __init__(self, config_path, msg_queue):
        super().__init__(config_path, msg_queue)
        self.without_input_queue = False
        self.id_map = defaultdict(int)
        self.infer_size = 0
        self.image_total = Manager().Value(c_uint64, 0)

    def init_self_args(self):
        super().init_self_args()

    def stop_handle(self, input_data):
        self.image_total.value = input_data.image_total

    def result_handle(self, input_data):
        if self.infer_res_save_path:
            name, _ = os.path.splitext(input_data.image_name)
            infer_res_name = f"infer_img_{name.split('_')[-1]}.txt"
            save_name = os.path.join(self.infer_res_save_path, infer_res_name)
            safe_list_writer(input_data.infer_result, save_name)
            log.info(f'save infer result to {infer_res_name} successfully')

        if input_data.image_id in self.id_map:
            self.id_map[input_data.image_id] -= len(input_data.infer_result)
            if not self.id_map[input_data.image_id]:
                self.id_map.pop(input_data.image_id)
                self.infer_size += 1
        else:
            remaining = input_data.sub_image_total - len(input_data.infer_result)
            if remaining:
                self.id_map[input_data.image_id] = remaining
            else:
                self.infer_size += 1

    def process(self, input_data):
        if isinstance(input_data, ProcessData):
            self.result_handle(input_data)
        elif isinstance(input_data, StopData):
            self.stop_handle(input_data)
        else:
            raise ValueError('unknown input data')

        if self.image_total.value and self.infer_size == self.image_total.value:
            self.send_to_next_module('stop')

    def stop(self):
        profiling_data = ProfilingData(module_name=self.module_name, instance_id=self.instance_id,
                                       device_id=self.device_id, process_cost_time=self.process_cost.value,
                                       send_cost_time=self.send_cost.value, image_total=self.image_total.value)
        self.msg_queue.put(profiling_data, block=False)
        self.is_stop = True
