#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
Description: Utils import manager
Author: MindX SDK
Create: 2022
History: NA
"""

from src.utils.common_utils import profiling, get_device_id
from src.utils.constant import NORMALIZE_MEAN, NORMALIZE_SCALE, NORMALIZE_STD, IMAGE_NET_IMAGE_MEAN, \
    IMAGE_NET_IMAGE_STD, MAX_PARALLEL_NUM, MIN_PARALLEL_NUM, MIN_DEVICE_ID, MAX_DEVICE_ID, TASK_QUEUE_SIZE, \
    DBNET_LIMIT_SIDE
from src.utils.cv_utils import get_hw_of_img, get_matched_gear_hw, padding_with_cv, normalize, to_chw_image, \
    expand, get_mini_boxes, unclip, construct_box, box_score_slow, get_rotate_crop_image, get_batch_list_greedy, \
    padding_batch, bgr_to_gray, array_to_texts, get_shape_from_gear, get_batch_from_gear, \
    resize_by_limit_max_side, box_score_fast, padding_with_np
from src.utils.logger import logger_instance as log
from src.utils.safe_utils import safe_list_writer, safe_div, safe_load_yaml, check_valid_dir, file_base_check, \
    check_valid_file, safe_img_read
