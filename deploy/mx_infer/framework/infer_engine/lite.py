from deploy.mx_infer.utils import check_valid_file
from .model_base import ModelBase


class LiteModel(ModelBase):
    def __init__(self, model_path, device_id):
        super().__init__()
        self.model_path = model_path
        self.device_id = device_id

        check_valid_file(model_path)
        self._init_model()

    def _init_model(self):
        import mindspore_lite as mslite

        context = mslite.Context()
        context.target = ['ascend']
        context.ascend.device_id = self.device_id

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

        self._input_shape = inputs[0].shape # shape before resize

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
        import os
        import yaml

        LITE_GEAR_FILE = 'LITE_GEAR_FILE'
        filepath = os.environ.get(LITE_GEAR_FILE, '')
        if not filepath:
            filepath = 'gear.yaml' if os.path.exists('gear.yaml') else ''

        if not filepath:
            raise ValueError(f"{LITE_GEAR_FILE} is empty.")

        with open(filepath) as f:
            content = f.read()
            content = content.replace("(", "[").replace(")", "]")
            content = yaml.safe_load(content)

        model_filename = os.path.basename(self.model_path)
        if model_filename not in content:
            raise ValueError(f"{model_filename} not in {filepath}.")

        attr = content.get(model_filename)
        attr = dict(attr[0], **attr[1])

        keys = {"shape", "gear"}
        if attr.keys() != keys:
            raise ValueError(f"Key {keys} must be in {filepath}.")

        if attr['shape'] != self.input_shape:
            raise ValueError(f"Shape of {self.model_path} is {self.input_shape}, "
                             f"but Shape of {model_filename} in {filepath} is {attr['shape']}.")
        gears = attr['gear']
        assert isinstance(gears, list)
        assert all(isinstance(gear, list) for gear in gears)
        assert all(len(gear) == 4 for gear in gears)

        return gears