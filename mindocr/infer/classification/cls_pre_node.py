import argparse
import os
import time
import sys
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))

from pipeline.framework.module_base import ModuleBase
from pipeline.tasks import TaskType
from .classification import ClsPreprocess


class ClsPreNode(ModuleBase):
    def __init__(self, args, msg_queue, tqdm_info):
        super(ClsPreNode, self).__init__(args, msg_queue, tqdm_info)
        self.cls_preprocesser = ClsPreprocess(args)
        self.task_type = self.args.task_type

    def init_self_args(self):
        super().init_self_args()
        return {"batch_size": 1}

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        if self.task_type.value == TaskType.REC.value:
            image = input_data.frame[0]
            data = [self.cls_preprocesser(image)["image"]]
            input_data.sub_image_size = 1
            input_data.data = data
            self.send_to_next_module(input_data)
        else:
            sub_image_list = input_data.sub_image_list
            data = [self.cls_preprocesser(split_image)["image"] for split_image in sub_image_list]
            input_data.data["cls_pre_res"] = data
            self.send_to_next_module(input_data)
