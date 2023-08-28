from typing import Dict, List, Tuple

import numpy as np

from ..core import Model, ShapeType
from ..data_process import build_postprocess, build_preprocess, cv_utils, gear_utils
from .infer_base import InferBase


class TextClassifier(InferBase):
    def __init__(self, args):
        super(TextClassifier, self).__init__(args)

    def _init_preprocess(self):
        self.preprocess_ops = build_preprocess(self.args.cls_config_path, self.requires_gear_hw)

    def _init_model(self):
        self.model = Model(
            backend=self.args.backend,
            device=self.args.device,
            model_path=self.args.cls_model_path,
            device_id=self.args.device_id,
        )

        shape_type, shape_value = self.model.get_shape_details()

        # Only check inputs[0] currently
        shape_value = shape_value[0]

        # check batch_size
        # assuming that the first dim is batch size
        batch_size = shape_value[0]

        batch_size = [batch_size] if not isinstance(batch_size, (tuple, list)) else batch_size

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

        self._hw_list = tuple(hw_list)
        self._bs_list = tuple(batch_size)

    def _init_postprocess(self):
        self.postprocess_ops = build_postprocess(self.args.cls_config_path)

    def get_params(self):
        return {"cls_batch_num": self._bs_list}

    def __call__(self, image: List[np.ndarray]) -> List:
        images = [image] if isinstance(image, np.ndarray) else image
        split_bs, split_data = self.preprocess(images)
        split_pred = [self.model_infer(data) for data in split_data]

        outputs = []
        for bs, pred in zip(split_bs, split_pred):
            output = self.postprocess_ops(pred, bs)
            outputs.append(output)

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
        return self.model.infer(data["net_inputs"])

    def postprocess(self, pred: List[np.ndarray], batch=-1) -> Dict[str, List]:
        pred = gear_utils.get_batch_from_padding(pred, batch)
        pred = pred[0] if len(pred) == 1 else tuple(pred)
        results = self.postprocess_ops(pred)  # {"angles": angles, "scores": scores}
        return results
