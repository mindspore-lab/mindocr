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
from shutil import rmtree

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../..')))

from deploy.infer_pipeline.pipeline import build_pipeline
from deploy.infer_pipeline.src.demo.python.src.utils import check_valid_dir, file_base_check
from tools.infer.text.args import get_args


def check_args(det_args):
    check_valid_dir(det_args.input_images_dir)
    file_base_check(det_args.config_path)
    if det_args.parallel_num < 1 or det_args.parallel_num > 4:
        raise ValueError(f'parallel num must between [1,4], current: {det_args.parallel_num}')


def save_path_init(path):
    if os.path.exists(path):
        rmtree(path)
    os.makedirs(path, 0o750)


def build_infer_pipeline(args):
    check_args(args)
    if args.pipeline_res_save_dir:
        save_path_init(args.pipeline_res_save_dir)
    build_pipeline(args)


if __name__ == '__main__':
    args = get_args()
    build_infer_pipeline(args)
