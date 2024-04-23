import os
import sys
import numpy as np
import yaml
from addict import Dict

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))

from pipeline.data_process.utils import cv_utils
from pipeline.utils import log
from pipeline.datatype import StopData
from pipeline.framework.module_base import ModuleBase


class DecodeNode(ModuleBase):
    def __init__(self, args, msg_queue, tqdm_info):
        super().__init__(args, msg_queue, tqdm_info)
        self.cost_time = 0
        self.avail_image_total = 0

    def init_self_args(self):
        super().init_self_args()

    def process(self, input_data):
        if isinstance(input_data, StopData):
            input_data.image_total = self.avail_image_total
            self.send_to_next_module(input_data)
            return

        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        # input contains np.ndarray, not need read again
        if len(input_data.frame) == len(input_data.image_path) and len(input_data.frame) > 0:
            self.avail_image_total += len(input_data.frame)
            self.send_to_next_module(input_data)
        else:
            img_read, img_path_read = [], []
            for image_path in input_data.image_path:
                try:
                    img_read.append(cv_utils.img_read(image_path))
                    img_path_read.append(image_path)
                    self.avail_image_total += 1
                except ValueError:
                    log.info(f"{image_path} is unavailable and skipped")
                    continue
            input_data.frame = img_read
            input_data.image_path = img_path_read
            self.send_to_next_module(input_data)
