import functools
from typing import Dict, List, Tuple, Union

import numpy as np

from ..core import Model, ShapeType
from ..data_process import build_postprocess, build_preprocess, gear_utils
from .infer_base import InferBase


class TextClassifier(InferBase):
    def __init__(self, args):
        super(TextClassifier, self).__init__(args)
        self._bs_list = []

    def init(self, warmup=False):
        self.model = Model(
            backend=self.args.backend, model_path=self.args.cls_model_path, device_id=self.args.device_id
        )
        shape_type, shape_info = self.model.get_shape_info()

        if shape_type not in (ShapeType.DYNAMIC_BATCHSIZE, ShapeType.STATIC_SHAPE):
            raise ValueError("Input shape must be static shape or dynamic batch_size for classification model.")

        if shape_type == ShapeType.DYNAMIC_BATCHSIZE:
            self._bs_list, _, model_height, model_width = shape_info
        else:
            batchsize, _, model_height, model_width = shape_info
            self._bs_list = [batchsize]

        preprocess_ops = build_preprocess(self.args.cls_config_path)
        self.preprocess_ops = functools.partial(preprocess_ops, target_size=(model_height, model_width))
        self.postprocess_ops = build_postprocess(self.args.cls_config_path)

        if warmup:
            self.model.warmup()

    def __call__(self, image: Union[np.ndarray, List[np.ndarray]]):
        images = [image] if isinstance(image, np.ndarray) else image
        split_bs, split_data = self.preprocess(images)
        split_pred = [self.model_infer(data) for data in split_data]

        outputs = []
        for bs, pred in zip(split_bs, split_pred):
            output = self.postprocess_ops(pred, bs)
            outputs.append(output)

        # TODO: merge outputs from different dynamic batch_size
        return outputs[0] if isinstance(image, np.ndarray) else outputs

    def preprocess(self, image: List[np.ndarray]) -> Tuple[List[int], List[Dict]]:
        num_image = len(image)
        batch_list = gear_utils.get_matched_gear_bs(num_image, self._bs_list)
        start_index = 0
        split_bs = []
        split_data = []
        for batch in batch_list:
            upper_bound = min(start_index + batch, num_image)
            split_input = image[start_index:upper_bound]
            split_output = self.preprocess_ops(split_input)
            split_output = gear_utils.padding_to_batch(split_output, batch)
            split_bs.append(upper_bound - start_index)
            split_data.append(split_output)
            start_index += batch

        return split_bs, split_data

    def model_infer(self, data: Dict) -> List[np.ndarray]:
        return self.model.infer([data["image"]])  # model infer for single input

    def postprocess(self, pred, batch=None) -> List:
        pred = gear_utils.get_batch_from_padding(pred, batch)
        return self.postprocess_ops(tuple(pred))  # [(label, score), ...]
