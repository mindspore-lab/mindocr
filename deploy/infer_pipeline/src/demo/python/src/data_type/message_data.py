#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
Description: Data types for transferring complex messages in message queue
Author: MindX SDK
Create: 2022
History: NA
"""

from dataclasses import dataclass


@dataclass
class StopSign:
    stop: bool = True


@dataclass
class ProfilingData:
    module_name: str = ''
    instance_id: int = ''
    device_id: int = 0
    process_cost_time: float = 0.
    send_cost_time: float = 0.
    image_total: int = -1
