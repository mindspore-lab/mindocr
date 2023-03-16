from mx_infer.framework import ModuleBase, ShapeType
from mx_infer.framework import Model
from mx_infer.utils import get_matched_gear_hw, padding_with_np


class DetInferProcess(ModuleBase):
    def __init__(self, args, msg_queue):
        super(DetInferProcess, self).__init__(args, msg_queue)
        self.without_input_queue = False
        self.model = None
        self.gear_list = None
        self.model_channel = None
        self.max_dot_gear = None

    def init_self_args(self):
        self.model = Model(engine_type=self.args.engine_type, model_path=self.args.det_model_path,
                           device_id=self.args.device_id)

        shape_type, shape_info = self.model.get_shape_info()
        if shape_type != ShapeType.DYNAMIC_IMAGESIZE:
            raise ValueError("Input shape must be dynamic image_size for detection model.")

        batchsize, channel, hw_list = shape_info
        if batchsize != 1:
            raise ValueError("Input batch size must be 1 for detection model.")

        self.gear_list = hw_list
        self.model_channel = channel
        self.max_dot_gear = max([(h, w) for h, w in hw_list], key=lambda x: x[0] * x[1])

        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return
        input_array = input_data.input_array
        _, _, h, w = input_array.shape

        matched_gear = get_matched_gear_hw((h, w), self.gear_list, self.max_dot_gear)
        input_array = padding_with_np(input_array, matched_gear)

        outputs = self.model.infer(input_array)
        output_array = outputs[0]

        output_array = output_array[:, :, :input_data.resize_h, :input_data.resize_w]

        # send the ready data to post module
        input_data.output_array = output_array
        input_data.input_array = None
        self.send_to_next_module(input_data)
