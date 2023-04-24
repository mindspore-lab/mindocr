import re

from .model_base import ModelBase
from ...utils import check_valid_file


class LiteModel(ModelBase):
    def __init__(self, model_path, device_id, precision_mode="fp32"):
        super().__init__()
        self.model_path = model_path
        self.device_id = device_id
        self.precision_mode = precision_mode

        check_valid_file(model_path)
        self._init_model()

    def _init_model(self):
        import mindspore_lite as mslite

        if mslite.__version__ < "2.0":
            raise ValueError(f"Only support mindspore lite >= 2.0, but got version {mslite.__version__}.")

        context = mslite.Context()
        context.target = ['ascend']
        context.ascend.device_id = self.device_id
        if self.precision_mode == "fp32":
            context.ascend.precision_mode = "enforce_fp32"
        else:
            context.ascend.precision_mode = "enforce_fp16"

        self.model = mslite.Model()
        self.model.build_from_file(self.model_path, mslite.ModelType.MINDIR, context)

        if not self.model:
            raise ValueError(f"The model file {self.model_path} load failed.")

        inputs = self.model.get_inputs()
        input_num = len(inputs)
        if input_num != 1:
            raise ValueError(f"Only support single input for model inference, "
                             f"but got {input_num} inputs for {self.model_path}.")

        if inputs[0].format != mslite.Format.NCHW:
            raise ValueError(f"Model inference only support NCHW format, "
                             f"but got {inputs[0].format.name} for {self.model_path}.")

        self._input_shape = inputs[0].shape  # shape before resize

    def infer(self, input):
        inputs = self.model.get_inputs()
        self.model.resize(inputs, [list(input.shape)])
        inputs[0].set_data_from_numpy(input)

        outputs = self.model.predict(inputs)
        outputs = [output.get_data_to_numpy().copy() for output in outputs]
        return outputs

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
