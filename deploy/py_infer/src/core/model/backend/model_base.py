import gc
from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np

from ....utils import check_valid_file


class ModelBase(metaclass=ABCMeta):
    """
    base class for model load and infer
    """

    def __init__(self, model_path: str, device: str, device_id: int):
        super().__init__()

        check_valid_file(model_path)

        self.model_path = model_path
        self.device = device
        self.device_id = device_id

        self._input_shape: list = []
        self._input_dtype: list = []
        self._input_num: int = 0

        self._init_model()

        assert len(self._input_dtype) == self._input_num
        assert len(self._input_shape) == self._input_num

    @abstractmethod
    def _init_model(self):
        pass

    @abstractmethod
    def infer(self, input: List[np.ndarray]) -> List[np.ndarray]:
        """
        model inference
        """
        pass

    @property
    def input_shape(self) -> List[List[int]]:
        """
        shape of inputs
        """
        return self._input_shape

    @property
    def input_dtype(self) -> List:
        return self._input_dtype

    @property
    def input_num(self) -> int:
        """
        number of Inputs
        """
        return self._input_num

    @abstractmethod
    def get_gear(self) -> List[List[int]]:
        """
        get gear for input_shape[0]
        """
        pass

    def __del__(self):
        if hasattr(self, "model") and self.model:
            del self.model
            gc.collect()
