import cv2

from ...framework import Model, ModuleBase


class CLSInferProcess(ModuleBase):
    def __init__(self, args, msg_queue):
        super(CLSInferProcess, self).__init__(args, msg_queue)
        self.model = None
        self.thresh = 0.9

    def init_self_args(self):
        self.model = Model(
            engine_type=self.args.engine_type, model_path=self.args.cls_model_path, device_id=self.args.device_id
        )

        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        input_array = input_data.input_array
        outputs = self.model.infer(input_array)
        output_array = outputs[0]

        for i in range(input_data.sub_image_size):
            if output_array[i, 1] > self.thresh:
                input_data.sub_image_list[i] = cv2.rotate(input_data.sub_image_list[i], cv2.ROTATE_180)

        input_data.input_array = None
        # send the ready data to post module
        self.send_to_next_module(input_data)
