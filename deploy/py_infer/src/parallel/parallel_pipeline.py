import argparse
import numpy as np
import os

import tqdm

from ..infer import TaskType
from ..data_process.utils import cv_utils
from .framework import ParallelPipelineManager
from .datatype import StopSign


class ParallelPipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.pipeline_manager = ParallelPipelineManager(args)
        self.input_queue = self.pipeline_manager.input_queue
        self.infer_params = {}

    def start_pipeline(self):
        self.pipeline_manager.start_pipeline()

    def stop_pipeline(self):
        self.pipeline_manager.stop_pipeline()

    def infer_for_images(self, input_images_dir):
        self.infer_params = dict(**self.pipeline_manager.module_params)
        self.send_image(input_images_dir)

    def send_image(self, images: str):
        """
        send image to input queue for pipeline
        """
        if not (os.path.isdir(images) or os.path.isfile(images)):
            raise ValueError("images must be a image path or dir.")

        # det, det(+cls)+rec
        batch_num = 1

        # cls, rec, layout
        if self.args.task_type in (TaskType.CLS, TaskType.REC):
            for name, value in self.infer_params.items():
                if name.endswith("_batch_num"):
                    batch_num = max(value)

        self._send_batch_image(images, batch_num)

    def _send_batch_image(self, images, batch_num):
        if os.path.isdir(images):
            show_progressbar = not self.args.show_log
            input_image_list = [os.path.join(images, path) for path in os.listdir(images) if not path.endswith(".txt")]

            images_num = len(input_image_list)
            for i in (
                tqdm.tqdm(range(images_num), desc="send image to pipeline") if show_progressbar else range(images_num)
            ):
                if i % batch_num == 0:
                    batch_images = input_image_list[i : i + batch_num]
                    self.input_queue.put(batch_images, block=True)
        else:
            self.input_queue.put([images], block=True)

    def infer_for_array(self, input_array):
        self.infer_params = dict(**self.pipeline_manager.module_params)
        self.send_array(input_array)

    def send_array(self, images):
        if isinstance(images, np.ndarray):
            self._send_batch_array([images], 1)
        elif isinstance(images, (tuple, list)):
            if len(images) == 0:
                return
            if not cv_utils.check_type_in_container(images, np.ndarray):
                ValueError(f"unknown input data, images should be np.ndarray, or tuple&list contain np.ndarray")
            # cls、rec
            batch_num = 1
            if self.args.task_type in (TaskType.CLS, TaskType.REC):
                for name, value in self.infer_params.items():
                    if name.endswith("_batch_num"):
                        batch_num = max(value)
            self._send_batch_array(images, batch_num)
        else:
            raise ValueError(f"unknown input data: {type(images)}")
            

    def _send_batch_array(self, images, batch_num):
        show_progressbar = not self.args.show_log
        images_num = len(images)
        for i in (
            tqdm.tqdm(range(images_num), desc="send image to pipeline") if show_progressbar else range(images_num)
        ):
            if i % batch_num == 0:
                batch_images = images[i : i + batch_num]
                self.input_queue.put(batch_images, block=True)