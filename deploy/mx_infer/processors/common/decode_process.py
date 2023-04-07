from mx_infer.framework import ModuleBase
from mx_infer.utils import safe_img_read, get_hw_of_img


class DecodeProcess(ModuleBase):
    def __init__(self, args, msg_queue):
        super().__init__(args, msg_queue)
        self.without_input_queue = False
        self.image_processor = None
        self.cost_time = 0

    def decode(self, image_path):
        image_src = safe_img_read(image_path)
        return image_src

    def init_self_args(self):
        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return
        image_path = input_data.image_path
        image_src = self.decode(image_path)
        h, w = get_hw_of_img(image_src)
        input_data.frame = image_src
        input_data.original_width = w
        input_data.original_height = h
        self.send_to_next_module(input_data)
