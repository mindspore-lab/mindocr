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
from .recognition import INFER_REC_MAP


class RecInferNode(ModuleBase):
    def __init__(self, args, msg_queue, tqdm_info):
        super(RecInferNode, self).__init__(args, msg_queue, tqdm_info)
        self.args = args
        self.rec_model = None
        self.task_type = self.args.task_type

    def init_self_args(self):
        with open(self.args.rec_model_name_or_config, "r") as f:
            self.yaml_cfg = Dict(yaml.safe_load(f))
        self.batch_size = self.yaml_cfg.predict.loader.batch_size
        RecModel = INFER_REC_MAP[self.yaml_cfg.predict.backend]
        self.rec_model = RecModel(self.args)
        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        data = input_data.data["rec_pre_res"]
        data = [np.expand_dims(d, 0) for d in data if len(d.shape) == 3]
        data = np.concatenate(data, axis=0)

        preds = []
        for batch_i in range(data.shape[0] // self.batch_size + 1):
            start_i = batch_i * self.batch_size
            end_i = (batch_i + 1) * self.batch_size
            d = data[start_i:end_i]
            if d.shape[0] == 0:
                continue
            pred = self.rec_model([d])
            preds.append(pred[0])
        preds = np.concatenate(preds, axis=0)
        input_data.data["rec_infer_res"] = {"pred": preds}
        self.send_to_next_module(input_data)
