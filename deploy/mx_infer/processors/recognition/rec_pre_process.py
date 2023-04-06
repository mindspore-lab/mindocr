import math
import os

import cv2
import numpy as np
from mindx.sdk import base

from mx_infer.data_type.process_data import ProcessData
from mx_infer.framework import ModuleBase, InferModelComb
from mx_infer.utils import get_batch_list_greedy, get_hw_of_img, safe_div, get_matched_gear_hw, \
    padding_with_cv, normalize, to_chw_image, expand, padding_batch, bgr_to_gray, get_shape_info, \
    check_valid_file, check_valid_dir, NORMALIZE_SCALE, NORMALIZE_MEAN, NORMALIZE_STD


class RecPreProcess(ModuleBase):
    def __init__(self, args, msg_queue):
        super(RecPreProcess, self).__init__(args, msg_queue)
        self.gear_list = []
        self.batchsize_list = []
        self.static_method = True
        self.model_height = 32
        self.model_channel = 3
        self.max_dot_gear = (0, 0)
        self.model_max_width = 4096
        self.model_min_width = 32
        self.scale = np.float32(NORMALIZE_SCALE)
        self.std = np.array(NORMALIZE_STD).astype(np.float32)
        self.mean = np.array(NORMALIZE_MEAN).astype(np.float32)
        self.task_type = args.task_type

    def get_shape_for_single_model(self, filename, device_id):
        check_valid_file(filename)
        model = base.model(filename, device_id)
        desc, shape_info = get_shape_info(model.input_shape(0), model.model_gear())
        del model

        if desc not in ("dynamic_shape", "dynamic_width"):
            error_info = f"static shape={shape_info}" if desc == "static_shape" \
                else f"optional {desc.split('_')} with gear"
            raise ValueError(
                f"input shape of model({filename}) must be dynamic shape without gear, "
                f"or have only optional width with gear, but got {error_info}.")

        batchsize, channel, height, width_list = shape_info

        if desc == "dynamic_shape":
            if batchsize != -1 or channel == -1 or height == -1 or width_list != -1:
                raise ValueError(
                    f"input shape must be only dynamic for batch_size and width if model({filename}) is dynamic shape "
                    f"without gear, but got shape={shape_info}.")

            self.static_method = False
            self.model_channel = channel
            self.model_height = height
        else:
            self.static_method = True
            self.batchsize_list.append(batchsize)
            self.model_channel = channel
            self.model_height = height
            self.gear_list = [(height, w) for w in width_list]

        return shape_info

    def init_self_args(self):
        device_id = self.args.device_id
        model_path = self.args.rec_model_path
        base.mx_init()

        if os.path.isfile(model_path):
            self.get_shape_for_single_model(model_path, device_id)

        if os.path.isdir(model_path):
            check_valid_dir(model_path)
            all_shape_info = []
            for path in os.listdir(model_path):
                shape_info = self.get_shape_for_single_model(os.path.join(model_path, path), device_id)
                all_shape_info.append(str((shape_info[1:])))
                if not self.static_method:
                    raise FileNotFoundError(
                        f"rec_model_dir({model_path}) must be file when use dynamic shape model without gear.")

            if len(set(all_shape_info)) != 1:
                raise ValueError(
                    f"input shape of all models in {model_path} must have same channel, height and optional width")

            if len(set(self.batchsize_list)) != len(self.batchsize_list):
                raise ValueError(f"input shape of all models in {model_path} must have different batch_size")

        if self.static_method:
            self.batchsize_list.sort()
            self.gear_list.sort()
            self.max_dot_gear = self.gear_list[-1]
            self.model_max_width = self.max_dot_gear[1]
            self.model_min_width = self.gear_list[0][1]
        else:
            self.model_max_width = math.floor(safe_div(self.model_max_width, 32)) * 32
            self.model_min_width = math.ceil(safe_div(self.model_min_width, 32)) * 32

        super().init_self_args()

    def get_max_width(self, image_list, max_wh_ratio):
        max_resize_w = 0
        max_width = max_wh_ratio * self.model_height
        for image in image_list:
            height, width = get_hw_of_img(image)
            ratio = safe_div(width, height)
            if math.ceil(ratio * self.model_height) > max_width:
                resize_w = max_width
            else:
                resize_w = math.ceil(self.model_height * ratio)
            max_resize_w = max(resize_w, max_resize_w)
            max_resize_w = max(min(max_resize_w, self.model_max_width), self.model_min_width)

        if self.static_method:
            _, gear_w = get_matched_gear_hw((self.model_height, max_resize_w), self.gear_list, self.max_dot_gear)
        else:
            gear_w = math.ceil(safe_div(max_resize_w, 32)) * 32
        return gear_w

    def preprocess(self, image_list, batchsize, max_resize_w, max_wh_ratio):
        input_list = []
        max_width = int(max_wh_ratio * self.model_height)
        for image in image_list:
            if self.model_channel == 1:
                image = bgr_to_gray(image)
            height, width = get_hw_of_img(image)
            ratio = safe_div(width, height)
            if math.ceil(ratio * self.model_height) > max_width:
                resize_w = max_width
            else:
                resize_w = math.ceil(self.model_height * ratio)
            resize_w = min(resize_w, self.model_max_width)
            crnn_image = cv2.resize(image, (resize_w, self.model_height))
            crnn_image = padding_with_cv(crnn_image, (self.model_height, max_resize_w))

            crnn_image = normalize(crnn_image, self.scale, self.std, self.mean)

            crnn_image = to_chw_image(crnn_image)

            input_list.append(crnn_image)

        input_array = expand(input_list)
        input_array = padding_batch(input_array, batchsize)
        return input_array

    def process(self, input_data):
        """
        split the sub image list to chunks by batch size and do the preprocess.
        If use dynamic model, the batch size will be the size of whole sub images list
        :param input_data: ProcessData
        :return: None
        """
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        if self.task_type == InferModelComb.REC:
            self.process_without_sub_image(input_data)
        else:
            self.process_with_sub_image(input_data)

    def process_without_sub_image(self, input_data):
        h, w = get_hw_of_img(input_data.frame)
        max_wh_ratio = safe_div(w, h)
        split_input = [input_data.frame]
        max_resize_w = self.get_max_width(split_input, max_wh_ratio)
        rec_model_inputs = self.preprocess(split_input, 1, max_resize_w, max_wh_ratio)

        send_data = ProcessData(sub_image_size=1,
                                image_path=input_data.image_path, image_total=input_data.image_total,
                                input_array=rec_model_inputs, frame=input_data.frame,
                                sub_image_total=1, image_name=input_data.image_name,
                                image_id=input_data.image_id)

        self.send_to_next_module(send_data)

    def process_with_sub_image(self, input_data):
        sub_image_list = input_data.sub_image_list
        infer_res_list = input_data.infer_result
        max_wh_ratio = input_data.max_wh_ratio
        batch_list = [input_data.sub_image_size]
        if self.static_method:
            batch_list = get_batch_list_greedy(input_data.sub_image_size, self.batchsize_list)

        start_index = 0
        for batch in batch_list:
            upper_bound = min(start_index + batch, input_data.sub_image_size)
            split_input = sub_image_list[start_index:upper_bound]
            split_infer_res = infer_res_list[start_index:upper_bound]
            max_resize_w = self.get_max_width(split_input, max_wh_ratio)
            rec_model_inputs = self.preprocess(split_input, batch, max_resize_w, max_wh_ratio)

            send_data = ProcessData(sub_image_size=min(upper_bound - start_index, batch),
                                    image_path=input_data.image_path, image_total=input_data.image_total,
                                    infer_result=split_infer_res, input_array=rec_model_inputs, frame=input_data.frame,
                                    sub_image_total=input_data.sub_image_total, image_name=input_data.image_name,
                                    image_id=input_data.image_id)

            start_index += batch
            self.send_to_next_module(send_data)
