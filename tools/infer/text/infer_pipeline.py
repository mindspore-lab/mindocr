#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
Description: OCR e2e infer demo based on PaddleOCR2.0 Server on Ascend device
Author: MindX SDK
Create: 2022
History: NA
"""

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../..')))

from tools.infer.text.args import get_args
import deploy.infer_pipeline.pipeline as pipeline

def main():
    args = get_args()
    pipeline.build_pipeline(args)

if __name__ == '__main__':
    main()
