import argparse
import gc
from abc import ABCMeta, abstractmethod

from ..core import Model


class InferBase(metaclass=ABCMeta):
    """
    base class for OCR inference
    """

    def __init__(self, args: argparse.Namespace, **kwargs):
        super().__init__()
        self.args = args

        self.model = None
        self._bs_list = []
        self._hw_list = []

    def init(self, *, preprocess=True, model=True, postprocess=True):
        if preprocess or model:
            self._init_model()

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

    def free_model(self):
        if hasattr(self, "model") and self.model:
            if isinstance(self.model, dict):
                self.model.clear()

            del self.model
            gc.collect()

            self.model = None

    def __del__(self):
        self.free_model()
