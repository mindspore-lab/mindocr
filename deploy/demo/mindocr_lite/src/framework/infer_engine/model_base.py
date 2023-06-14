from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np


class ModelBase(metaclass=ABCMeta):
    """
    base class for model load and infer
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = None

    @abstractmethod
    def _init_model(self):
        pass

    @abstractmethod
    def infer(self, input: np.ndarray) -> List[np.ndarray]:
        """
        model inference, just for single input
        """
        pass

    @property
    @abstractmethod
    def input_shape(self) -> List[int]:
        """
        get input_shape[0]
        """
        pass

    @abstractmethod
    def get_gear(self) -> List[List[int]]:
        """
        get gear for input_shape[0]
        """
        pass

    def __del__(self):
        if hasattr(self, "model") and self.model:
            del self.model
