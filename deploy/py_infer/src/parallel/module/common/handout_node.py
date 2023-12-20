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

    def process(self, input_mix_data):
        if isinstance(input_mix_data, StopSign):
            data = self.process_stop_sign()
            self.send_to_next_module(data)
        elif isinstance(input_mix_data, np.ndarray):
            input_data, info_data = input_mix_data
            data = self.process_image_array([input_data])
            data.task_images_num = info_data[0]
            data.taskid = info_data[1]
            data.data_type = 1
            self.send_to_next_module(data)
        elif isinstance(input_mix_data, (tuple, list)):
            input_data, info_data = input_mix_data
            if len(input_data) == 0:
                return
            if cv_utils.check_type_in_container(input_data, str):
                data = self.process_image_path(input_data)
                data.data_type = 0
            elif cv_utils.check_type_in_container(input_data, np.ndarray):
                data = self.process_image_array(input_data)
                data.data_type = 1
            else:
                raise ValueError(
                    "unknown input data, input_data should be StopSign, or tuple&list contains str or np.ndarray"
                )
            data.task_images_num = info_data[0]
            data.taskid = info_data[1]
            self.send_to_next_module(data)
        else:
            raise ValueError(f"unknown input data: {type(input_mix_data)}")

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
            if self.args.input_array_save_dir:
                image_path = os.path.join(self.args.input_array_save_dir, f"input_array_{self.image_total}.jpg")
                if len(image_array_list[i].shape) != 3:
                    log.info(f"image_array_list[{i}] array with shape {image_array_list[i].shape} is invalid")
                    continue
                try:
                    cv_utils.img_write(image_path, image_array_list[i])
                except cv2.error:
                    log.info(f"image_array_list[{i}] with shape {image_array_list[i].shape} array is invalid")
                    continue
                log.info(f"sending array(saved at {image_path}) to pipleine")
                array_save_path.append(image_path)
            else:
                array_save_path.append(str(i))
            frames.append(image_array_list[i])

            self.image_total += 1
        data = ProcessData(frame=frames, image_path=array_save_path)
        return data

    def process_stop_sign(self):
        # image_total of StopData will be assigned in decode_node
        return StopData(skip=True)
