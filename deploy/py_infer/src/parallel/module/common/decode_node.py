from ....data_process.utils import cv_utils
from ...framework import ModuleBase


class DecodeNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super().__init__(args, msg_queue)
        self.cost_time = 0

    def init_self_args(self):
        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        input_data.frame = [cv_utils.img_read(image_path) for image_path in input_data.image_path]

        self.send_to_next_module(input_data)
