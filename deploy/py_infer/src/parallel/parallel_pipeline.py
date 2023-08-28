import argparse
import os

import tqdm

from ..infer import TaskType
from .framework import ParallelPipelineManager


class ParallelPipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.pipeline_manager = ParallelPipelineManager(args)
        self.input_queue = self.pipeline_manager.input_queue
        self.infer_params = {}

    def infer_for_images(self, input_images_dir):
        self.pipeline_manager.start_pipeline()
        self.infer_params = dict(**self.pipeline_manager.module_params)
        self.send_image(input_images_dir)
        self.pipeline_manager.stop_pipeline()

    def send_image(self, images: str):
        """
        send image to input queue for pipeline
        """
        if not (os.path.isdir(images) or os.path.isfile(images)):
            raise ValueError("images must be a image path or dir.")

        # det、det(+cls)+rec
        batch_num = 1

        # cls、rec
        if self.args.task_type in (TaskType.CLS, TaskType.REC):
            for name, value in self.infer_params.items():
                if name.endswith("_batch_num"):
                    batch_num = max(value)

        self._send_batch_image(images, batch_num)

    def _send_batch_image(self, images, batch_num):
        if os.path.isdir(images):
            show_progressbar = not self.args.show_log
            input_image_list = [os.path.join(images, path) for path in os.listdir(images)]

            images_num = len(input_image_list)
            for i in (
                tqdm.tqdm(range(images_num), desc="send image to pipeline") if show_progressbar else range(images_num)
            ):
                if i % batch_num == 0:
                    batch_images = input_image_list[i : i + batch_num]
                    self.input_queue.put(batch_images, block=True)
        else:
            self.input_queue.put([images], block=True)
