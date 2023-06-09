import numpy as np

from ...utils import check_valid_file, suppress_stdout
from .model_base import ModelBase


class MindXModel(ModelBase):
    def __init__(self, model_path, device_id, precision_mode="fp32"):
        super().__init__()
        self.model_path = model_path
        self.device_id = device_id
        self.precision_mode = precision_mode

        check_valid_file(model_path)
        self._init_model()

    def _init_model(self):
        global base, Tensor
        with suppress_stdout():
            from mindx.sdk import Tensor, base, visionDataFormat

        base.mx_init()

        self.model = base.model(self.model_path, self.device_id)
        if not self.model:
            raise ValueError(f"The model file {self.model_path} load failed.")

        if self.model.input_format != visionDataFormat.NCHW:
            raise ValueError(
                f"Model inference only support NCHW format, "
                f"but got {self.model.input_format.name} for {self.model_path}."
            )

    def infer(self, input):
        input = Tensor(input)
        outputs = self.model.infer(input)
        list([output.to_host() for output in outputs])
        outputs = [np.array(output) for output in outputs]
        return outputs

    @property
    def input_shape(self):
        return self.model.input_shape(0)

    def get_gear(self):
        gears = self.model.model_gear()

        # dynamic shape or static shape
        if not gears:
            return gears

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
