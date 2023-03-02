#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
Description: Data Classes for Init Module
Author: MindX SDK
Create: 2022
History: NA
"""

from dataclasses import dataclass, field
from enum import Enum


class InferModelComb(Enum):
    DET = 1  # Detection Model
    REC = 2  # Recognition Model
    CLS_DET = 3  # Classifier And Detection Model
    CLS_REC = 4  # Classifier And Recognition Model
    DET_REC = 5  # Detection And Recognition Model
    CLS_DET_REC = 6  # Classifier And Detection And Recognition Model


class ConnectType(Enum):
    MODULE_CONNECT_ONE = 0
    MODULE_CONNECT_CHANNEL = 1
    MODULE_CONNECT_PAIR = 2
    MODULE_CONNECT_RANDOM = 3


@dataclass
class ModuleOutputInfo:
    module_name: str
    connect_type: ConnectType
    output_queue_list_size: int
    output_queue_list: list = field(default_factory=lambda: [])


@dataclass
class ModuleInitArgs:
    pipeline_name: str
    module_name: str
    instance_id: -1


@dataclass
class ModuleDesc:
    module_name: str
    module_count: int


@dataclass
class ModuleConnectDesc:
    module_send_name: str
    module_recv_name: str
    connect_type: ConnectType = field(default_factory=lambda: ConnectType.MODULE_CONNECT_RANDOM)


@dataclass
class ModulesInfo:
    module_list: list = field(default_factory=lambda: [])
    input_queue_list: list = field(default_factory=lambda: [])
