import numpy as np
from mindx.sdk import base, Tensor

from mx_infer.framework import ModuleBase
from mx_infer.utils import get_matched_gear_hw, padding_with_np, \
    get_shape_info


class DetInferProcess(ModuleBase):
    def __init__(self, args, msg_queue):
        super(DetInferProcess, self).__init__(args, msg_queue)
        self.without_input_queue = False
        self.model = None
        self.gear_list = None
        self.model_channel = None
        self.max_dot_gear = None

    def init_self_args(self):
        device_id = self.args.device_id
        model_path = self.args.det_model_path

        base.mx_init()
        self.model = base.model(model_path, device_id)

        desc, shape_info = get_shape_info(self.model.input_shape(0), self.model.model_gear())
        if desc != "dynamic_height_width":
            raise ValueError("model input shape must be dynamic image_size with gear.")

        _, channel, hw_list = shape_info
        self.gear_list = hw_list
        self.model_channel = channel
        self.max_dot_gear = max([(h, w) for h, w in hw_list], key=lambda x: x[0] * x[1])

        self.warmup()
        super().init_self_args()

    def warmup(self):
        dummy_tensor = np.random.randn(1, self.model_channel, self.max_dot_gear[0], self.max_dot_gear[1]).astype(
            np.float32)
        inputs = [Tensor(dummy_tensor)]
        self.model.infer(inputs)

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return
        input_array = input_data.input_array
        _, _, h, w = input_array.shape

        matched_gear = get_matched_gear_hw((h, w), self.gear_list, self.max_dot_gear)
        input_array = padding_with_np(input_array, matched_gear)
        inputs = [Tensor(input_array)]
        output = self.model.infer(inputs)
        if not output:
            output = self.model.infer(inputs)
        output_array = output[0]
        output_array.to_host()
        output_array = np.array(output_array)

        output_array = output_array[:, :, :input_data.resize_h, :input_data.resize_w]

        # send the ready data to post module
        input_data.output_array = output_array
        input_data.input_array = None
        self.send_to_next_module(input_data)
