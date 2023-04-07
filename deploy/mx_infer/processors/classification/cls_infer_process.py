import os

import cv2
import numpy as np
from mindx.sdk import base, Tensor

from mx_infer.framework import ModuleBase
from mx_infer.utils import check_valid_file


class CLSInferProcess(ModuleBase):
    def __init__(self, args, msg_queue):
        super(CLSInferProcess, self).__init__(args, msg_queue)
        self.model = None
        self.static_method = True
        self.thresh = 0.9

    def init_self_args(self):
        device_id = self.args.device_id
        model_path = self.args.cls_model_path

        base.mx_init()
        if model_path and os.path.isfile(model_path):
            check_valid_file(model_path)
            self.model = base.model(model_path, device_id)
        else:
            raise FileNotFoundError('cls model path must be a file')

        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        input_array = input_data.input_array
        inputs = [Tensor(input_array)]
        output = self.model.infer(inputs)
        output_array = output[0]
        output_array.to_host()
        output_array = np.array(output_array)

        for i in range(input_data.sub_image_size):
            if output_array[i, 1] > self.thresh:
                input_data.sub_image_list[i] = cv2.rotate(input_data.sub_image_list[i],
                                                          cv2.ROTATE_180)

        input_data.input_array = None
        # send the ready data to post module
        self.send_to_next_module(input_data)
