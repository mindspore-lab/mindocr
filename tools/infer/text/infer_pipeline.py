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
import time
from collections import defaultdict
from multiprocessing import Process, Queue
from shutil import rmtree

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../..')))

from args import get_args
from deploy.infer_pipeline.src.demo.python.src.data_type import StopSign
from deploy.infer_pipeline.src.demo.python.src.framework import ModuleDesc, ModuleConnectDesc, ModuleManager
from deploy.infer_pipeline.src.demo.python.src.processors import MODEL_DICT
from deploy.infer_pipeline.src.demo.python.src.utils import log, safe_load_yaml, profiling, safe_div, check_valid_dir, file_base_check, TASK_QUEUE_SIZE


def save_path_init(path):
    if os.path.exists(path):
        rmtree(path)
    os.makedirs(path, 0o750)


def image_sender(images_path, send_queue):
    input_image_list = [os.path.join(images_path, path) for path in os.listdir(images_path)]
    for image_path in input_image_list:
        send_queue.put(image_path, block=True)


def build_pipeline(config_path, parallel_num, input_queue, pipeline_res_save_dir):
    # config_dict = safe_load_yaml(config_path)
    module_desc_list = [ModuleDesc('HandoutProcess', 1), ModuleDesc('DecodeProcess', parallel_num), ]

    module_order = config_dict.get('module_order', None)
    if module_order is None:
        raise ValueError('cannot find the order of module')

    for model_name in module_order:
        model_name = model_name.lower()
        if model_name not in MODEL_DICT:
            log.error(f'unsupported model {model_name}')
            raise ValueError(f'unsupported model {model_name}')
        for name, count in MODEL_DICT.get(model_name, []):
            module_desc_list.append(ModuleDesc(name, count * parallel_num))

    module_desc_list.append(ModuleDesc('CollectProcess', 1))
    module_connect_desc_list = []
    for i in range(len(module_desc_list) - 1):
        module_connect_desc_list.append(ModuleConnectDesc(module_desc_list[i].module_name,
                                                          module_desc_list[i + 1].module_name))

    module_size = sum(desc.module_count for desc in module_desc_list)
    log.info(f'module_size: {module_size}')
    msg_queue = Queue(module_size)

    manager = ModuleManager(msg_queue, input_queue, config_path, pipeline_res_save_dir)
    manager.register_modules(str(os.getpid()), module_desc_list, 1)
    manager.register_module_connects(str(os.getpid()), module_connect_desc_list)

    # start the pipeline, init start
    manager.run_pipeline()

    # waiting for task receive
    while not msg_queue.full():
        continue

    start_time = time.time()
    # release all init sign
    for _ in range(module_size):
        msg_queue.get()

    # release the stop sign, infer start
    manager.stop_manager.get(block=False)

    manager.deinit_pipeline_module()

    cost_time = time.time() - start_time

    # collect the profiling data
    profiling_data = defaultdict(lambda: [0, 0])
    image_total = 0
    for _ in range(module_size):
        msg_info = msg_queue.get()
        profiling_data[msg_info.module_name][0] += msg_info.process_cost_time
        profiling_data[msg_info.module_name][1] += msg_info.send_cost_time
        if msg_info.module_name != -1:
            image_total = msg_info.image_total

    profiling(profiling_data, image_total)

    log.info(f'total cost {cost_time:.2f}s, FPS: {safe_div(image_total, cost_time):.2f}')
    msg_queue.close()
    msg_queue.join_thread()


def check_args(opts):
    check_valid_dir(opts.input_images_dir)
    if opts.parallel_num < 1 or opts.parallel_num > 4:
        raise ValueError(f'parallel num must between [1,4], current: {opts.parallel_num}')


if __name__ == '__main__':
    args = get_args()
    check_args(args)
    if args.pipeline_res_save_dir:
        save_path_init(args.pipeline_res_save_dir)

    task_queue = Queue(TASK_QUEUE_SIZE)
    process = Process(target=build_pipeline, args=(args.config_path, args.parallel_num, task_queue,
                                                   args.pipeline_res_save_dir))
    process.start()

    image_sender(images_path=args.input_images_dir, send_queue=task_queue)
    task_queue.put(StopSign(), block=True)
    process.join()
    process.close()
