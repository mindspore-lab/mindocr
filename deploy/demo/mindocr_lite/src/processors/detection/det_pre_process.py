from ...framework import Model, ModuleBase, ShapeType
from ...operators import build_preprocess
from ...utils import get_hw_of_img, get_matched_gear_hw


class DetPreProcess(ModuleBase):
    def __init__(self, args, msg_queue):
        super(DetPreProcess, self).__init__(args, msg_queue)
        self.gear_list = None
        self.max_dot_gear = None

    def init_self_args(self):
        model = Model(
            engine_type=self.args.engine_type, model_path=self.args.det_model_path, device_id=self.args.device_id
        )

        shape_type, shape_info = model.get_shape_info()
        del model

        if shape_type not in (ShapeType.DYNAMIC_IMAGESIZE, ShapeType.STATIC_SHAPE):
            raise ValueError("Input shape must be static shape or dynamic image_size for detection model.")

        if shape_type == ShapeType.DYNAMIC_IMAGESIZE:
            batchsize, _, hw_list = shape_info
        else:
            batchsize, _, h, w = shape_info
            hw_list = [(h, w)]

        if batchsize != 1:
            raise ValueError("Input batch size must be 1 for detection model.")

        self.gear_list = hw_list
        self.max_dot_gear = hw_list[-1]

        self.preprocess = build_preprocess(self.args.det_algorithm)
        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return
        image = input_data.frame

        resized_params = {
            "Resize": {"dst_hw": get_matched_gear_hw(get_hw_of_img(image), self.gear_list, self.max_dot_gear)}
        }

        dst_image = self.preprocess(image, resized_params)

        # send the ready data to infer module
        input_data.input_array = dst_image

        self.send_to_next_module(input_data)
