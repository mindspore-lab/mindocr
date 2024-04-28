import os
import sys

import numpy as np
import time

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))

from pipeline.framework.module_base import ModuleBase
from pipeline.tasks import TaskType
from .layout import LayoutPreprocess

class LayoutPreNode(ModuleBase):
    def __init__(self, args, msg_queue, tqdm_info):
        super(LayoutPreNode, self).__init__(args, msg_queue, tqdm_info)
        self.layout_preprocesser = LayoutPreprocess(args)
        self.task_type = self.args.task_type

    def init_self_args(self):
        super().init_self_args()
        return {"batch_size": 1}

    def process(self, input_data):
        """
        Input:
          - input_data.frame: [np.ndarray], shape:[-1,-1,3], shape e.g. [792,601,3]
        Output:
          - input_data.data["image"]: np.ndarray, shape:[1,3,800,800]
          - input_data.data["raw_img_shape"]: (int, int), value e.g. (792,601)
          - input_data.data["target_size"]: [int, int], value e.g. (800,800)
          - input_data.data["image_ids"]: int, value e.g. 0
          - input_data.data["hw_ori"]: (int, int), value e.g. (792,601)
          - input_data.data["hw_scale"]: np.ndarray, shape:[1,2], value e.g. (1.0101,1.3311)
          - input_data.data["pad"]: np.ndarray, shape:[1,2], value e.g. (4,99.5)
        """
        if input_data.skip:
            self.send_to_next_module(input_data)
            return
        if len(input_data.frame) == 0:
            return

        image = input_data.frame[0]  # bs = 1 for layout
        data = {
            "image": image,
            "raw_img_shape": image.shape[:2],
            "target_size": [800, 800],
        }
        data = self.layout_preprocesser(data)
        # print(data)

        if len(data["image"].shape) == 3:
            data["image"] = np.expand_dims(data["image"], 0)

        # if self.task_type.value == TaskType.LAYOUT.value and not (self.args.crop_save_dir or self.args.vis_layout_save_dir):
            # input_data.frame = None

        input_data.data = data

        self.send_to_next_module(input_data)
