import os
import sys
import logging

import mindspore as ms
import numpy as np

from addict import Dict

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))


from tools.infer.text.utils import get_ckpt_file
from mindocr.models.builder import build_model
from mindocr.data.transforms import create_transforms, run_transforms
from mindocr.utils.logger import set_logger
from mindocr.postprocess import build_postprocess

from typing import List
import yaml

import copy

import time

from mindocr.infer.utils.model import MSModel, LiteModel

class LayoutPreprocess(object):
    def __init__(self, args) -> None:
        self.args = args
        with open(args.layout_model_name_or_config, "r") as f:
            self.yaml_cfg = Dict(yaml.safe_load(f))
        
        for transform in self.yaml_cfg.predict.dataset.transform_pipeline:
            if "DecodeImage" in transform:
                transform["DecodeImage"].update({"keep_ori": True})
                break
            if "func_name" in transform:
                func_name = transform.pop("func_name")
                args = copy.copy(transform)
                transform.clear()
                transform[func_name] = args
        self.transforms = create_transforms(self.yaml_cfg.predict.dataset.transform_pipeline)

    def __call__(self, data):
        data = run_transforms(data, self.transforms)
        return data


algo_to_model_name = {
    "YOLOV8": "layout_yolov8n",
}
logger = logging.getLogger("mindocr")


class LayoutModelMS(MSModel):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = args
        self.model_name = algo_to_model_name[args.layout_algorithm]
        self.config_path = args.layout_config_path
        self._init_model(self.model_name, self.config_path)


class LayoutModelLite(LiteModel):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = args
        self.model_name = algo_to_model_name[args.layout_algorithm]
        self.config_path = args.layout_config_path
        self._init_model(self.model_name, self.config_path)


class LayoutPostProcess(object):
    def __init__(self, args) -> None:
        self.args = args
        with open(args.layout_model_name_or_config, "r") as f:
            self.yaml_cfg = Dict(yaml.safe_load(f))
        self.transforms = build_postprocess(self.yaml_cfg.postprocess)

        self.meta_data_indices = self.yaml_cfg.predict.dataset.pop("meta_data_column_index", None)


    def __call__(self, pred, img_shape, meta_info):
        return self.transforms(pred, img_shape, meta_info=meta_info)

INFER_LAYOUT_MAP = {"MindSporeLite": LayoutModelLite, "MindSpore": LayoutModelMS}