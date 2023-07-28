from typing import List

import numpy as np

from ....utils import suppress_stdout
from .model_base import ModelBase


class MindXModel(ModelBase):
    def __init__(self, model_path, device, device_id):
        if device.lower() != "ascend":
            raise ValueError(f"ACL inference only support Ascend device, but got {device}.")

        super().__init__(model_path, device, device_id)

    def _init_model(self):
        global base, Tensor
        with suppress_stdout():
            from mindx.sdk import Tensor, base

        base.mx_init()

        self.model = base.model(self.model_path, self.device_id)
        if not self.model:
            raise ValueError(f"The model file {self.model_path} load failed.")

        # dynamic batch size/image size name: ascend_mbatch_shape_data
        # dynamic aipp name: ascend_dynamic_aipp_data
        # TODO: self._input_num remove dynamic aipp input_num 1.
        self._input_num = self.model.input_num - 1 if self.model.model_gear() else self.model.input_num
        self._input_shape = [self.model.input_shape(i) for i in range(self._input_num)]
        self._input_dtype = [self.__dtype_to_nptype(self.model.input_dtype(i)) for i in range(self._input_num)]

    def infer(self, inputs: List[np.ndarray]):
        inputs = [Tensor(input) for input in inputs]
        outputs = self.model.infer(inputs)
        list([output.to_host() for output in outputs])
        outputs = [np.array(output) for output in outputs]
        return outputs

    def get_gear(self):
        gears = self.model.model_gear()

        # TODO: shape gear don't support for multi input
        if self._input_num > 1 and gears:
            raise ValueError(
                f"Shape gear don‘t support model input_num > 1 currently, \
                but got input_num = {self._input_num} for {self.model_path}!"
            )

        # dynamic shape or static shape
        if not gears:
            return gears

        # TODO: only support NCHW format for shape gear
        # dynamic_dims
        if len(gears[0]) == 4:
            return gears

        # dynamic_batch_size
        if len(gears[0]) == 1:
            chw = self.input_shape[1:]
            return [gear + chw for gear in gears]

        # dynamic_image_size
        if len(gears[0]) == 2:
            nc = self.input_shape[:2]
            return [nc + gear for gear in gears]

        raise ValueError(f"Get gear value failed for {self.model_path}. Please Check ATC conversion process!")

    def __dtype_to_nptype(self, type_):
        dtype = base.dtype

        return {
            dtype.bool: np.bool_,
            dtype.int8: np.int8,
            dtype.int16: np.int16,
            dtype.int32: np.int32,
            dtype.int64: np.int64,
            dtype.uint8: np.uint8,
            dtype.uint16: np.uint16,
            dtype.uint32: np.uint32,
            dtype.uint64: np.uint64,
            dtype.float16: np.float16,
            dtype.float32: np.float32,
            dtype.double: np.float64,
        }[type_]
