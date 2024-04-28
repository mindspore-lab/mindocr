import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))

from pipeline.framework.module_base import ModuleBase
from pipeline.tasks import TaskType
from .layout import INFER_LAYOUT_MAP

import time
import numpy as np

import yaml

from addict import Dict

class LayoutInferNode(ModuleBase):
    def __init__(self, args, msg_queue, tqdm_info):
        super(LayoutInferNode, self).__init__(args, msg_queue, tqdm_info)
        self.args = args
        self.layout_model = None
        self.task_type = self.args.task_type
        self.i = 0

    def init_self_args(self):
        with open(self.args.layout_model_name_or_config, "r") as f:
            self.yaml_cfg = Dict(yaml.safe_load(f))
        LayoutModel = INFER_LAYOUT_MAP[self.yaml_cfg.predict.backend]
        self.layout_model = LayoutModel(self.args)
        super().init_self_args()

    def process(self, input_data):
        """
        Input:
          - input_data.data["image"]: np.ndarray, shape:[1,3,800,800]
        Output:
          - input_data.data["pred"]: [np.ndarray], shape:[1,?,?], shape e.g. [1,13294, 9] (note:[bs, N, 5+nc])
          - input_data.data["img_shape"]: (int, int, int, int), value e.g. (1,3,800,800)
        """
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        data = input_data.data["image"]
        pred = self.layout_model([data])

        input_data.data["pred"] = pred
        input_data.data["img_shape"] = input_data.data["image"].shape

        self.send_to_next_module(input_data)