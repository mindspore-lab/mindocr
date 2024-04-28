import os
import sys

import numpy as np
import time

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))

from pipeline.framework.module_base import ModuleBase
from pipeline.tasks import TaskType
from .detection import DetPreprocess

class DetPreNode(ModuleBase):
    def __init__(self, args, msg_queue, tqdm_info):
        super(DetPreNode, self).__init__(args, msg_queue, tqdm_info)
        self.det_preprocesser = DetPreprocess(args)
        self.task_type = self.args.task_type

    def init_self_args(self):
        super().init_self_args()
        return {"batch_size": 1}

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        if self.task_type.value in (TaskType.DET.value, TaskType.DET_REC.value, TaskType.DET_CLS_REC.value):
            if len(input_data.frame) == 0:
                return
            image = input_data.frame[0]  # bs = 1 for det
        else:
            if len(input_data.data["layout_images"]) == 0:
                return
            image = input_data.data["layout_images"][0]  # bs = 1 for det
        data = self.det_preprocesser({"image": image})
        # print(data)
        
        if len(data["image"].shape) == 3:
            data["image"] = np.expand_dims(data["image"], 0)
        data["shape_list"] = np.expand_dims(data["shape_list"], 0)
        # print(data["image"].shape)
        # time.sleep(1000)
        if self.task_type.value == TaskType.DET.value and not (self.args.crop_save_dir or self.args.vis_det_save_dir):
            input_data.frame = None

        if self.task_type.value in (TaskType.LAYOUT_DET.value, TaskType.LAYOUT_DET_REC, TaskType.LAYOUT_DET_CLS_REC):
            input_data.data["det_pre_res"] = data
        else:
            input_data.data = {"det_pre_res": data}

        self.send_to_next_module(input_data)
