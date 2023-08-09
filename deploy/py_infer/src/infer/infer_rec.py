import os
from typing import Dict, List, Tuple

import numpy as np

from ..core import Model, ShapeType
from ..data_process import build_postprocess, build_preprocess, cv_utils, gear_utils
from .infer_base import InferBase


class TextRecognizer(InferBase):
    def __init__(self, args):
        super(TextRecognizer, self).__init__(args)
        self.model: Dict[int, Model] = {}

    def __load_model(self, filename):
        model = Model(
            backend=self.args.backend, device=self.args.device, model_path=filename, device_id=self.args.device_id
        )
        shape_type, shape_value = model.get_shape_details()

        # Only check inputs[0] currently
        shape_value = shape_value[0]

        # check batch_size
        # assuming that the first dim is batch size
        batch_size = shape_value[0]
        if not isinstance(batch_size, (tuple, list)):
            self.model[batch_size] = model
            batch_size = [batch_size]
        else:
            for n in batch_size:
                self.model[n] = model

        # check h/w
        if shape_type == ShapeType.DYNAMIC_SHAPE:
            # without any checks, and assuming that h/w is dynamic
            # if not, may throw exceptions in model_infer for un-matched h/w
            hw_list = []
        elif shape_type == ShapeType.DYNAMIC_IMAGESIZE:  # Only NCHW
            *_, hw_list = shape_value
        else:  # static shape or dynamic batch size
            if len(shape_value) == 4:  # only NCHW
                *_, h, w = shape_value
                hw_list = [(h, w)]
            else:
                hw_list = []  # without any checks

        self._bs_list += tuple(batch_size)
        self._hw_list += tuple(hw_list)

        return shape_type, shape_value

    def _init_preprocess(self):
        self.preprocess_ops = build_preprocess(self.args.rec_config_path, support_gear=self.requires_gear_hw)

    def _init_model(self):
        model_path = self.args.rec_model_path

        if os.path.isfile(model_path):
            self.__load_model(model_path)

        if os.path.isdir(model_path):
            chw_list = []
            info = (
                "If --rec_model_path is folder, every model in the folder must be dynamic image_size when converting "
                "model, they must have same candidate image_size list and different batch_size. For example: "
                "NCHW format, model1_shape = (1,3,-1,-1) and model2_shape = (8,3,-1,-1), batch_size has different "
                "values of 1 and 8, and '--dynamic_dims' for image_size must have same value for every "
                "model when converting to the mindspore lite MindIR model or OM model."
            )
            for path in os.listdir(model_path):
                shape_type, shape_value = self.__load_model(os.path.join(model_path, path))
                chw_list.append(str((shape_value[2:])))  # [[c,h,w], ...]
                if shape_type != ShapeType.DYNAMIC_IMAGESIZE:
                    raise ValueError(info + f" But found that {path} is not a dynamic image_size model.")
            if len(set(chw_list)) != 1 or len(set(self._bs_list)) != len(self._bs_list):
                raise ValueError(info + f" Please check every model file in {model_path}.")

    def _init_postprocess(self):
        params = {"character_dict_path": self.args.character_dict_path}
        self.postprocess_ops = build_postprocess(self.args.rec_config_path, **params)

    def get_params(self):
        return {"rec_batch_num": self._bs_list}

    def __call__(self, images: List[np.ndarray]) -> List:
        split_bs, split_data = self.preprocess(images)
        split_pred = [self.model_infer(data) for data in split_data]
        outputs = []
        for bs, pred in zip(split_bs, split_pred):
            output = self.postprocess(pred, bs)
            outputs.extend(output)

        # TODO: merge outputs from different dynamic batch_size
        return outputs

    def preprocess(self, image: List[np.ndarray]) -> Tuple[List[int], List[Dict]]:
        num_image = len(image)
        batch_list = gear_utils.get_matched_gear_bs(num_image, self._bs_list)
        start_index = 0
        split_bs = []
        split_data = []
        for batch in batch_list:
            upper_bound = min(start_index + batch, num_image)
            split_input = image[start_index:upper_bound]

            if self.requires_gear_hw:
                target_size = self._get_batch_matched_hw(cv_utils.get_batch_hw_of_img(split_input))
                split_output = self.preprocess_ops(split_input, target_size=target_size)
            else:
                split_output = self.preprocess_ops(split_input)

            if self.requires_gear_bs:
                split_output = gear_utils.padding_to_batch(split_output, batch)

            split_bs.append(upper_bound - start_index)
            split_data.append(split_output)
            start_index += batch

        return split_bs, split_data

    def model_infer(self, data: Dict) -> List[np.ndarray]:
        net_inputs = data["net_inputs"]
        bs, *_ = net_inputs[0].shape
        n = bs if bs in self.model else -1
        return self.model[n].infer(net_inputs)

    def postprocess(self, pred: List[np.ndarray], batch=None):
        pred = gear_utils.get_batch_from_padding(pred, batch)
        pred = pred[0] if len(pred) == 1 else tuple(pred)
        return self.postprocess_ops(pred)
