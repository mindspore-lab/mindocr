import os
from collections import defaultdict
from ctypes import c_uint64
from multiprocessing import Manager

from abc import ABCMeta, abstractmethod
import sys
import numpy as np
import yaml
from addict import Dict

import logging

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))

from tools.infer.text.utils import get_ckpt_file
from mindocr.models.builder import build_model
from typing import List

logger = logging.getLogger("mindocr")

class BaseModel(metaclass=ABCMeta):
    def __init__(self, args) -> None:
        self.model = None
        self.args = args
        self.pretrained = True
        self.ckpt_load_path = ""
        self.amp_level = "O0"
    
    @abstractmethod
    def __call__(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        pass

    @abstractmethod
    def _init_model(self, model_name, config_path):
        pass


class MSModel(BaseModel):
    def __init__(self, args) -> None:
        super().__init__(args)
    
    def _init_model(self, model_name, config_path):
        global ms
        import mindspore as ms
        
        self.config_path = config_path
        with open(self.config_path, "r") as f:
            self.yaml_cfg = Dict(yaml.safe_load(f))
        self.ckpt_load_path = self.yaml_cfg.predict.ckpt_load_path
        if self.ckpt_load_path is None:
            self.pretrained = True
            self.ckpt_load_path = None
        else:
            self.pretrained = False
            self.ckpt_load_path = get_ckpt_file(self.ckpt_load_path)
        
        ms.set_context(device_target=self.yaml_cfg.predict.get("device_target", "Ascend"))
        ms.set_context(device_id=self.yaml_cfg.predict.get("device_id", 0))
        ms.set_context(mode=self.yaml_cfg.predict.get("mode", 0))
        if self.yaml_cfg.system.get("distribute", False):
            ms.communication.init()
            ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
        if self.yaml_cfg.predict.get("max_device_memory", None):
            ms.set_context(max_device_memory=self.yaml_cfg.predict.get("max_device_memory"))
        self.amp_level = self.yaml_cfg.predict.get("amp_level", "O0")
        if ms.get_context("device_target") == "GPU" and self.amp_level == "O3":
            logger.warning(
                "Model prediction does not support amp_level O3 on GPU currently."
                "The program has switched to amp_level O2 automatically."
            )
            self.amp_level = "O2"
        self.model = build_model(
            model_name,
            ckpt_load_path=self.ckpt_load_path,
            amp_level=self.amp_level,
        )
        self.model.set_train(False)
        logger.info(
            "Init mindspore model: {}. Model weights loaded from {}".format(
                model_name, "pretrained url" if self.pretrained else self.ckpt_load_path
            )
        )
    def __call__(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        input_ms = [ms.Tensor.from_numpy(input) for input in inputs]
        output = self.model(*input_ms)
        outputs = [output.asnumpy()]
        return outputs


class LiteModel(BaseModel):
    def __init__(self, args) -> None:
        super().__init__(args)
    
    def _init_model(self, model_name, config_path):
        global mslite
        import mindspore_lite as mslite
        self.config_path = config_path
        with open(self.config_path, "r") as f:
            self.yaml_cfg = Dict(yaml.safe_load(f))
        self.ckpt_load_path = self.yaml_cfg.predict.ckpt_load_path
        context = mslite.Context()
        device_target = self.yaml_cfg.predict.get("device_target", "Ascend")
        context.target = [device_target.lower()]
        if device_target.lower() == "ascend":
            context.ascend.device_id = self.yaml_cfg.predict.get("device_id", 0)
        elif device_target.lower() == "gpu":
            context.gpu.device_id = self.yaml_cfg.predict.get("device_id", 0)
        else:
            pass
        self.model = mslite.Model()
        self.model.build_from_file(self.ckpt_load_path, mslite.ModelType.MINDIR, context)

    def __call__(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        model_inputs = self.model.get_inputs()
        inputs_shape = [list(input.shape) for input in inputs]
        self.model.resize(model_inputs, inputs_shape)
        for i, input in enumerate(inputs):
            model_inputs[i].set_data_from_numpy(input)
        model_outputs = self.model.predict(model_inputs)
        outputs = [output.get_data_to_numpy().copy() for output in model_outputs]
        return outputs
