import argparse
import gc
from abc import ABCMeta, abstractmethod
from functools import lru_cache
from typing import Tuple

from ..core import Model
from ..data_process import gear_utils


class InferBase(metaclass=ABCMeta):
    """
    base class for OCR inference
    """

    def __init__(self, args: argparse.Namespace, **kwargs):
        super().__init__()
        self.args = args

        self.model = None
        self.requires_gear_hw = False
        self.requires_gear_bs = False

        self._bs_list: Tuple[int] = tuple()
        self._hw_list: Tuple[Tuple[int]] = tuple()

    def init(self, *, preprocess=True, model=True, postprocess=True):
        if preprocess or model:
            self._init_model()

            if len(self._hw_list) > 0:
                self.requires_gear_hw = True

            if len(self._bs_list) > 1 or self._bs_list[0] not in {-1, 1}:
                self.requires_gear_bs = True  # need padding to batch
            if self._bs_list[0] == -1:  # when dynamic shape, len(self._bs_list[0]) == 1, self._bs_list[0] == -1
                batch_size_map = {
                    "TextDetector": 1,
                    "TextClassifier": self.args.cls_batch_num,
                    "TextRecognizer": self.args.rec_batch_num,
                }
                self._bs_list = (batch_size_map[self.__class__.__name__],)

            self._bs_list = tuple(sorted(self._bs_list))
            self._hw_list = tuple(sorted(self._hw_list, key=lambda x: x[0] * x[1]))

        if preprocess:
            self._init_preprocess()

        if postprocess:
            self._init_postprocess()

        if not model:
            self.free_model()

        if model:
            if isinstance(self.model, dict):
                for _model in self.model.values():
                    _model.warmup()
            elif isinstance(self.model, Model):
                self.model.warmup()
            else:
                pass

    @abstractmethod
    def _init_preprocess(self):
        pass

    @abstractmethod
    def _init_model(self):
        pass

    @abstractmethod
    def _init_postprocess(self):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        pass

    @abstractmethod
    def model_infer(self, *args, **kwargs):
        pass

    @abstractmethod
    def postprocess(self, *args, **kwargs):
        pass

    @lru_cache()
    def _get_batch_matched_hw(self, img_hw_list: Tuple[Tuple[int]]) -> Tuple[int]:
        resized_hw_list = [gear_utils.get_matched_gear_hw(hw, self._hw_list) for hw in img_hw_list]
        max_hw = max(resized_hw_list, key=lambda x: x[0] * x[1])

        return max_hw

    def free_model(self):
        if hasattr(self, "model") and self.model:
            if isinstance(self.model, dict):
                self.model.clear()

            del self.model
            gc.collect()

            self.model = None

    def __del__(self):
        self.free_model()
