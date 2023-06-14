import argparse
from abc import ABCMeta, abstractmethod


class InferBase(metaclass=ABCMeta):
    """
    base class for OCR inference
    """

    def __init__(self, args: argparse.Namespace, **kwargs):
        super().__init__()
        self.args = args
        self.model = None

    @abstractmethod
    def init(self, **kwargs):
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
            if isinstance(self.model, (tuple, list)):
                for model in self.model:
                    del model
            else:
                del self.model
        self.model = None

    def __del__(self):
        self.free_model()
