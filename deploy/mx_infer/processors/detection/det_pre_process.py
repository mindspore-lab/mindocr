import numpy as np

from mx_infer.framework import ModuleBase
from mx_infer.utils import normalize, to_chw_image, \
    expand, get_hw_of_img, resize_by_limit_max_side, IMAGE_NET_IMAGE_STD, IMAGE_NET_IMAGE_MEAN, DBNET_LIMIT_SIDE, \
    NORMALIZE_SCALE


class DetPreProcess(ModuleBase):
    def __init__(self, args, msg_queue):
        super(DetPreProcess, self).__init__(args, msg_queue)
        self.without_input_queue = False
        self.gear_list = []
        self.max_dot_gear = (0, 0)
        self.scale = np.float32(NORMALIZE_SCALE)
        self.std = np.array(IMAGE_NET_IMAGE_STD).reshape((1, 1, 3)).astype(np.float32)
        self.mean = np.array(IMAGE_NET_IMAGE_MEAN).reshape((1, 1, 3)).astype(np.float32)
        self.model_channel = 3

    def init_self_args(self):
        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return
        image = input_data.frame

        # resize image by the limit side
        dst_image = resize_by_limit_max_side(image, DBNET_LIMIT_SIDE)
        resize_h, resize_w = get_hw_of_img(image_src=dst_image)

        # normalize by scale std mean
        dst_image = normalize(dst_image, self.scale, self.std, self.mean)

        # make memory contiguous by NCHW format
        dst_image = to_chw_image(dst_image)

        input_array = expand(dst_image)
        # send the ready data to infer module
        input_data.input_array = input_array
        input_data.resize_h = resize_h
        input_data.resize_w = resize_w

        self.send_to_next_module(input_data)
