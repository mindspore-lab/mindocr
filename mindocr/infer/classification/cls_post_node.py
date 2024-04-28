import os
import time
import sys
import numpy as np
import yaml
from addict import Dict

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))

import cv2
import numpy as np

from pipeline.framework.module_base import ModuleBase
from pipeline.tasks import TaskType
from .classification import ClsPostprocess
from tools.infer.text.utils import crop_text_region
from pipeline.data_process.utils.cv_utils import crop_box_from_image


class ClsPostNode(ModuleBase):
    def __init__(self, args, msg_queue, tqdm_info):
        super(ClsPostNode, self).__init__(args, msg_queue, tqdm_info)
        self.cls_postprocess = ClsPostprocess(args)
        self.task_type = self.args.task_type
        self.cls_thresh = 0.9

    def init_self_args(self):
        super().init_self_args()

    def process(self, input_data):
        """
        Input:
          - input_data.data: [np.ndarray], shape:[?,2]
        Output:
          - input_data.sub_image_list: [np.ndarray], shape:[1,3,-1,-1], e.g. [1,3,48,192]
          - input_data.data = None
          or
          - input_data.infer_result = [(str, float)]
        """
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        data = input_data.data["cls_infer_res"]
        pred = data["pred"]
        output = self.cls_postprocess(pred)
        angles = output["angles"]
        scores = np.array(output["scores"]).tolist()

        batch = input_data.sub_image_size
        if self.task_type.value in (TaskType.DET_CLS_REC.value, TaskType.Layout_DET_CLS_REC.value):
            sub_images = input_data.sub_image_list
            for i in range(batch):
                angle, score = angles[i], scores[i]
                if "180" == angle and score > self.cls_thresh:
                    sub_images[i] = cv2.rotate(sub_images[i], cv2.ROTATE_180)
            input_data.sub_image_list = sub_images
        else:
            input_data.infer_result = [(angle, score) for angle, score in zip(angles, scores)]

        self.send_to_next_module(input_data)
