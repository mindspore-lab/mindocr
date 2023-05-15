import re
from typing import List

import numpy as np

from .model_base import ModelBase
from ....utils import check_valid_file


class LiteModel(ModelBase):
    def __new__(cls, *args, **kwargs):
        from mindspore_lite.version import __version__

        if __version__ < "2.0":
            return super(LiteModel, cls).__new__(_LiteModelV1)
        else:
            return super(LiteModel, cls).__new__(_LiteModelV2)

    def __init__(self, model_path, device_id):
        super().__init__()
        self.model_path = model_path
        self.device_id = device_id

        self._input_shape = None
        check_valid_file(model_path)
        self._init_model()

    @property
    def input_shape(self):
        return self._input_shape

    def get_gear(self):
        gears = []

        # MSLite does not provide API to get gear value, so we parse it from origin file.
        with open(self.model_path, 'rb') as f:
            content = f.read()

        matched = re.search(rb"_all_origin_gears_inputs.*?\xa0", content, flags=re.S)
        if not matched:
            return gears

        matched_text = matched.group()
        shape_text = re.findall(rb"(?<=:4:)\d+,\d+,\d+,\d+", matched_text)

        if not shape_text:
            raise ValueError(f"Get gear value failed for {self.model_path}. "
                             f"Please Check converter_lite conversion process!")

        for text in shape_text:
            gear = [int(x) for x in text.decode(encoding='utf-8').split(",")]
            gears.append(gear)

        return gears


class _LiteModelV1(LiteModel):
    def _init_model(self):
        import mindspore_lite as mslite

        ascend_device_info = mslite.AscendDeviceInfo(device_id=self.device_id)
        context = mslite.Context()
        context.append_device_info(ascend_device_info)

        self.model = mslite.Model()
        self.model.build_from_file(self.model_path, mslite.ModelType.MINDIR, context)

        inputs = self.model.get_inputs()
        input_num = len(inputs)
        if input_num != 1:
            raise ValueError(f"Only support single input for model inference, "
                             f"but got {input_num} inputs for {self.model_path}.")

        if inputs[0].get_format() != mslite.Format.NCHW:
            raise ValueError(f"Model inference only support NCHW format, "
                             f"but got {inputs[0].format.name} for {self.model_path}.")

        self._input_shape = inputs[0].get_shape()  # shape before resize

    def infer(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        model_inputs = self.model.get_inputs()
        inputs_shape = [list(input.shape) for input in inputs]
        self.model.resize(model_inputs, inputs_shape)

        for i, input in enumerate(inputs):
            model_inputs[i].set_data_from_numpy(input)

        model_outputs = self.model.get_outputs()
        self.model.predict(model_inputs, model_outputs)

        outputs = [output.get_data_to_numpy().copy() for output in model_outputs]
        return outputs


class _LiteModelV2(LiteModel):
    def _init_model(self):
        import mindspore_lite as mslite

        context = mslite.Context()
        context.target = ['ascend']
        context.ascend.device_id = self.device_id

        self.model = mslite.Model()
        self.model.build_from_file(self.model_path, mslite.ModelType.MINDIR, context)

        inputs = self.model.get_inputs()
        input_num = len(inputs)
        if input_num != 1:
            raise ValueError(f"Only support single input for model inference, "
                             f"but got {input_num} inputs for {self.model_path}.")

        if inputs[0].format != mslite.Format.NCHW:
            raise ValueError(f"Model inference only support NCHW format, "
                             f"but got {inputs[0].format.name} for {self.model_path}.")

        self._input_shape = inputs[0].shape  # shape before resize

    def infer(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        model_inputs = self.model.get_inputs()
        inputs_shape = [list(input.shape) for input in inputs]
        self.model.resize(model_inputs, inputs_shape)

        for i, input in enumerate(inputs):
            model_inputs[i].set_data_from_numpy(input)

        model_outputs = self.model.predict(model_inputs)
        outputs = [output.get_data_to_numpy().copy() for output in model_outputs]
        return outputs
