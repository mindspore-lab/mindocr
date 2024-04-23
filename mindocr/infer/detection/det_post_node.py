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
from .detection import DetPostprocess
from tools.infer.text.utils import crop_text_region
from pipeline.data_process.utils.cv_utils import crop_box_from_image


class DetPostNode(ModuleBase):
    def __init__(self, args, msg_queue, tqdm_info):
        super(DetPostNode, self).__init__(args, msg_queue, tqdm_info)
        self.det_postprocess = DetPostprocess(args)
        self.task_type = self.args.task_type

    def init_self_args(self):
        super().init_self_args()


    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        pred = input_data.data["pred"][0]

        data_dict = {"shape_list": input_data.data["shape_list"]}
        boxes = self.det_postprocess(pred, data_dict)

        boxes = boxes["polys"][0]
        infer_res_list = []
        for box in boxes:
            infer_res_list.append(box.tolist())

        input_data.infer_result = infer_res_list

        if self.task_type.value in (TaskType.DET_REC.value, TaskType.DET_CLS_REC.value):
            input_data.sub_image_total = len(infer_res_list)
            input_data.sub_image_size = len(infer_res_list)

            image = input_data.frame[0]  # bs=1 for det
            sub_image_list = []
            for box in infer_res_list:
                sub_image = crop_box_from_image(image, np.array(box))
                sub_image_list.append(sub_image)
            input_data.sub_image_list = sub_image_list

        input_data.data = None

        if not (self.args.crop_save_dir or self.args.vis_det_save_dir or self.args.vis_pipeline_save_dir):
            input_data.frame = None

        if not infer_res_list:
            input_data.skip = True

        self.send_to_next_module(input_data)
