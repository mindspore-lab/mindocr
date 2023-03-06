#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
Description: import manager of CRNN / SVTR (CTC decode base model)
Author: MindX SDK
Create: 2022
History: NA
"""

from .rec_infer_process import RecInferProcess
from .rec_post_process import RecPostProcess
from .rec_pre_process import RecPreProcess

SUPPORT_REC_MODEL = ['CRNN', 'SVTR']
