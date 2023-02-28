#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
Description: Common utility
Author: MindX SDK
Create: 2022
History: NA
"""

from src.utils.constant import MIN_DEVICE_ID, MAX_DEVICE_ID
from src.utils.logger import logger_instance as log
from src.utils.safe_utils import safe_div


def profiling(profiling_data, image_total):
    e2e_cost_time_per_image = 0
    for module_name in profiling_data:
        data = profiling_data[module_name]
        total_time = data[0]
        process_time = data[0] - data[1]
        send_time = data[1]
        process_avg = safe_div(process_time * 1000, image_total)
        e2e_cost_time_per_image += process_avg
        log.info(f'{module_name} cost total {total_time:.2f} s, process avg cost {process_avg:.2f} ms, '
                 f'send waiting time avg cost {safe_div(send_time * 1000, image_total):.2f} ms')
        log.info('----------------------------------------------------')
    log.info(f'e2e cost time per image {e2e_cost_time_per_image}ms')


def get_device_id(config, model_name):
    model_config = config.get(model_name, {})
    if model_config and model_config.get('device_id', None) is not None:
        device_id = model_config.get('device_id', None)
    else:
        device_id = config.get('device_id', 0)

    if isinstance(device_id, list):
        if any(device_id_ < MIN_DEVICE_ID or device_id_ > MAX_DEVICE_ID for device_id_ in device_id):
            raise ValueError(f'device id must in [0,8], current setting is {device_id}')
    else:
        if device_id < MIN_DEVICE_ID or device_id > MAX_DEVICE_ID:
            raise ValueError(f'device id must in [0,8], current setting is {device_id}')
    return device_id
