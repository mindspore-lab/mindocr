import logging
import os
import time
import sys
import numpy as np
import yaml
from addict import Dict
from typing import List

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))

from tools.infer.text.utils import get_ckpt_file
from mindocr.data.transforms import create_transforms, run_transforms
from mindocr.postprocess import build_postprocess
from mindocr.infer.utils.model import MSModel, LiteModel


algo_to_model_name = {
    "MV3": "cls_mobilenet_v3_small_100_model",
}
logger = logging.getLogger("mindocr")

class ClsPreprocess(object):
    def __init__(self, args):
        self.args = args
        with open(args.cls_model_name_or_config, "r") as f:
            self.yaml_cfg = Dict(yaml.safe_load(f))
        self.transforms = create_transforms(self.yaml_cfg.predict.dataset.transform_pipeline)
    
    def __call__(self, img):
        print(img)
        data = {"image": img}
        data = run_transforms(data, self.transforms[1:])
        return data


class ClsModelMS(MSModel):
    def __init__(self, args):
        self.args = args
        self.model_name = algo_to_model_name[args.cls_algorithm]
        self.config_path = args.cls_config_path
        self._init_model(self.model_name, self.config_path)


class ClsModelLite(LiteModel):
    def __init__(self, args):
        self.args = args
        self.model_name = algo_to_model_name[args.cls_algorithm]
        self.config_path = args.cls_config_path
        self._init_model(self.model_name, self.config_path)

INFER_CLS_MAP = {"MindSporeLite": ClsModelLite, "MindSpore": ClsModelMS}

class ClsPostprocess(object):
    def __init__(self, args):
        self.args = args
        with open(args.cls_model_name_or_config, "r") as f:
            self.yaml_cfg = Dict(yaml.safe_load(f))        
        self.postprocessor = build_postprocess(self.yaml_cfg.postprocess)
    
    def __call__(self, pred):
        return self.postprocessor(pred)