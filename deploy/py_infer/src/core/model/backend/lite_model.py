import re
from typing import List

import numpy as np

from .model_base import ModelBase


class LiteModel(ModelBase):
    def __init__(self, model_path, device, device_id):
        from mindspore_lite.version import __version__

        if __version__ < "2.0":
            raise ValueError(f"Only support mindspore lite >= 2.0, but got version {__version__}.")

        super().__init__(model_path, device, device_id)

    def _init_model(self):
        global mslite
        import mindspore_lite as mslite

        context = mslite.Context()
        context.target = [self.device.lower()]

        if self.device.lower() == "ascend":
            context.ascend.device_id = self.device_id
        elif self.device.lower() == "gpu":
            context.gpu.device_id = self.device_id
        else:
            pass

        self.model = mslite.Model()
        self.model.build_from_file(self.model_path, mslite.ModelType.MINDIR, context)

        inputs = self.model.get_inputs()
        self._input_num = len(inputs)
        self._input_shape = [x.shape for x in inputs]  # shape before resize
        self._input_dtype = [self.__dtype_to_nptype(x.dtype) for x in inputs]

    def infer(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        model_inputs = self.model.get_inputs()
        inputs_shape = [list(input.shape) for input in inputs]
        self.model.resize(model_inputs, inputs_shape)

        for i, input in enumerate(inputs):
            model_inputs[i].set_data_from_numpy(input)

        model_outputs = self.model.predict(model_inputs)
        outputs = [output.get_data_to_numpy().copy() for output in model_outputs]
        return outputs

    def get_gear(self):
        # Only support shape gear for Ascend device.
        if self.device.lower() != "ascend":
            return []

        gears = []

        # MSLite does not provide API to get gear value, so we parse it from origin file.
        with open(self.model_path, "rb") as f:
            content = f.read()

        matched = re.search(rb"_all_origin_gears_inputs.*?\xa0", content, flags=re.S)

        # TODO: shape gear don't support for multi input
        if self._input_num > 1 and matched:
            raise ValueError(
                f"Shape gear don‘t support model input_num > 1 currently, \
                but got input_num = {self._input_num} for {self.model_path}!"
            )

        if not matched:
            return gears

        # TODO: only support NCHW format for shape gear
        matched_text = matched.group()
        shape_text = re.findall(rb"(?<=:4:)\d+,\d+,\d+,\d+", matched_text)

        if not shape_text:
            raise ValueError(
                f"Get gear value failed for {self.model_path}. Please Check converter_lite conversion process!"
            )

        for text in shape_text:
            gear = [int(x) for x in text.decode(encoding="utf-8").split(",")]
            gears.append(gear)

        return gears

    def __dtype_to_nptype(self, type_):
        DataType = mslite.DataType

        return {
            DataType.BOOL: np.bool_,
            DataType.INT8: np.int8,
            DataType.INT16: np.int16,
            DataType.INT32: np.int32,
            DataType.INT64: np.int64,
            DataType.UINT8: np.uint8,
            DataType.UINT16: np.uint16,
            DataType.UINT32: np.uint32,
            DataType.UINT64: np.uint64,
            DataType.FLOAT16: np.float16,
            DataType.FLOAT32: np.float32,
            DataType.FLOAT64: np.float64,
        }[type_]
