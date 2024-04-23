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
    "DB": "dbnet_resnet50",
    "DB++": "dbnetpp_resnet50",
    "DB_MV3": "dbnet_mobilenetv3",
    "DB_PPOCRv3": "dbnet_ppocrv3",
    "PSE": "psenet_resnet152",
}
logger = logging.getLogger("mindocr")

class DetPreprocess(object):
    def __init__(self, args):
        self.args = args
        with open(args.det_model_name_or_config, "r") as f:
            self.yaml_cfg = Dict(yaml.safe_load(f))
        for transform in self.yaml_cfg.predict.dataset.transform_pipeline:
            if "DecodeImage" in transform:
                transform["DecodeImage"].update({"keep_ori": True})
                break
        self.transforms = create_transforms(self.yaml_cfg.predict.dataset.transform_pipeline)
    
    def __call__(self, img):
        data = {"image": img}
        data = run_transforms(data, self.transforms[1:])
        return data


class DetModelMS(MSModel):
    def __init__(self, args):
        self.args = args
        self.model_name = algo_to_model_name[args.det_algorithm]
        self.config_path = args.det_config_path
        self._init_model(self.model_name, self.config_path)


class DetModelLite(LiteModel):
    def __init__(self, args):
        self.args = args
        self.model_name = algo_to_model_name[args.det_algorithm]
        self.config_path = args.det_config_path
        self._init_model(self.model_name, self.config_path)

INFER_DET_MAP = {"MindSporeLite": DetModelLite, "MindSpore": DetModelMS}


class DetPostprocess(object):
    def __init__(self, args):
        self.args = args
        with open(args.det_model_name_or_config, "r") as f:
            self.yaml_cfg = Dict(yaml.safe_load(f))        
        self.transforms = build_postprocess(self.yaml_cfg.postprocess)
    
    def __call__(self, pred, data):
        return self.transforms(pred, **data)