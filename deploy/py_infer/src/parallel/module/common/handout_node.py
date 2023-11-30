import os

import cv2
import numpy as np

from ....data_process.utils import cv_utils
from ....utils import log
from ...datatype import ProcessData, StopData, StopSign
from ...framework.module_base import ModuleBase


class HandoutNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super().__init__(args, msg_queue)
        self.image_total = 0

    def init_self_args(self):
        super().init_self_args()

    def process(self, input_data):
        if isinstance(input_data, StopSign):
            data = self.process_stop_sign()
            self.send_to_next_module(data)
        elif isinstance(input_data, np.ndarray):
            data = self.process_image_array([input_data])
            self.send_to_next_module(data)
        elif isinstance(input_data, (tuple, list)):
            if len(input_data) == 0:
                return
            if cv_utils.check_type_in_container(input_data, str):
                data = self.process_image_path(input_data)
            elif cv_utils.check_type_in_container(input_data, np.ndarray):
                data = self.process_image_array(input_data)
            else:
                ValueError(
                    "unknown input data, input_data should be StopSign, or tuple&list contains str or np.ndarray"
                )
            self.send_to_next_module(data)
        else:
            raise ValueError(f"unknown input data: {type(input_data)}")

    def process_image_path(self, image_path_list):
        """
        image_folder: List[str], path to images
        """
        log.info(f"sending {', '.join([os.path.basename(x) for x in image_path_list])} to pipleine")
        data = ProcessData(image_path=image_path_list)
        self.image_total += len(image_path_list)
        return data

    def process_image_array(self, image_array_list):
        """
        image_array_list: List[np.ndarray], array of images
        """
        frames = []
        array_save_path = []
        image_num = len(image_array_list)
        for i in range(image_num):
            image_path = os.path.join(self.args.input_array_save_dir, f"input_array_{self.image_total}.jpg")
            if len(image_array_list[i].shape) != 3:
                log.info(f"image_array_list[{i}] with shape {image_array_list[i].shape} is invalid")
                continue
            try:
                cv_utils.img_write(image_path, image_array_list[i])
            except cv2.error:
                log.info(f"Failed to write image_array_list[{i}] with shape {image_array_list[i].shape}")
                continue
            log.info(f"sending array(saved at {image_path}) to pipleine")
            frames.append(image_array_list[i])
            array_save_path.append(image_path)
            self.image_total += 1
        data = ProcessData(frame=frames, image_path=array_save_path)
        return data

    def process_stop_sign(self):
        # image_total of StopData will be assigned in decode_node
        return StopData(skip=True)
