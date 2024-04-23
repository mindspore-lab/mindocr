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
from .recognition import RecPostprocess
from tools.infer.text.utils import crop_text_region
from pipeline.data_process.utils.cv_utils import crop_box_from_image


class RecPostNode(ModuleBase):
    def __init__(self, args, msg_queue, tqdm_info):
        super(RecPostNode, self).__init__(args, msg_queue, tqdm_info)
        self.rec_postprocess = RecPostprocess(args)
        self.task_type = self.args.task_type

    def init_self_args(self):
        super().init_self_args()


    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        data = input_data.data
        pred = data["pred"]
        output = self.rec_postprocess(pred)
        texts = output["texts"]
        confs = output["confs"]
        if self.task_type.value == TaskType.REC.value:
            input_data.infer_result = output["texts"]
        else:
            for i, (text, conf) in enumerate(zip(texts, confs)):
                input_data.infer_result[i].append(text)
                input_data.infer_result[i].append(conf)
        input_data.data = None
        self.send_to_next_module(input_data)
