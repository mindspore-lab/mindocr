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
    "CRNN": "crnn_resnet34",
    "RARE": "rare_resnet34",
    "CRNN_CH": "crnn_resnet34_ch",
    "RARE_CH": "rare_resnet34_ch",
    "SVTR": "svtr_tiny",
    "SVTR_PPOCRv3_CH": "svtr_ppocrv3_ch",
}
logger = logging.getLogger("mindocr")

class RecPreprocess(object):
    def __init__(self, args):
        self.args = args
        with open(args.rec_model_name_or_config, "r") as f:
            self.yaml_cfg = Dict(yaml.safe_load(f))
        self.transforms = create_transforms(self.yaml_cfg.predict.dataset.transform_pipeline)
    
    def __call__(self, img):
        data = {"image": img}
        # ZHQ TODO: [1:] ???
        data = run_transforms(data, self.transforms[1:])
        return data


class RecModelMS(MSModel):
    def __init__(self, args):
        self.args = args
        self.model_name = algo_to_model_name[args.rec_algorithm]
        self.config_path = args.rec_config_path
        self._init_model(self.model_name, self.config_path)


class RecModelLite(LiteModel):
    def __init__(self, args):
        self.args = args
        self.model_name = algo_to_model_name[args.rec_algorithm]
        self.config_path = args.rec_config_path
        self._init_model(self.model_name, self.config_path)

INFER_REC_MAP = {"MindSporeLite": RecModelLite, "MindSpore": RecModelMS}

class RecPostprocess(object):
    def __init__(self, args):
        self.args = args
        with open(args.rec_model_name_or_config, "r") as f:
            self.yaml_cfg = Dict(yaml.safe_load(f))        
        self.postprocessor = build_postprocess(self.yaml_cfg.postprocess)
    
    def __call__(self, pred):
        return self.postprocessor(pred)