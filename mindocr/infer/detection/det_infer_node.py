import os
import time
import sys
import numpy as np
import yaml
from addict import Dict

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))

from pipeline.framework.module_base import ModuleBase
from pipeline.tasks import TaskType
from .detection import INFER_DET_MAP


class DetInferNode(ModuleBase):
    def __init__(self, args, msg_queue, tqdm_info):
        super(DetInferNode, self).__init__(args, msg_queue, tqdm_info)
        self.args = args
        self.det_model = None
        self.task_type = self.args.task_type

    def init_self_args(self):
        with open(self.args.det_model_name_or_config, "r") as f:
            self.yaml_cfg = Dict(yaml.safe_load(f))
        DetModel = INFER_DET_MAP[self.yaml_cfg.predict.backend]
        self.det_model = DetModel(self.args)
        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        data = input_data.data["image"]
        pred = self.det_model([data])

        input_data.data = {"pred": pred, "shape_list": input_data.data["shape_list"]}

        self.send_to_next_module(input_data)
