import functools
from typing import Union, List, Tuple

import numpy as np

from .infer_base import InferBase
from ..data_process import gear_utils, build_preprocess, build_postprocess
from ..core import Model, ShapeType


class TextClassifier(InferBase):
    def __init__(self, args):
        super(TextClassifier, self).__init__(args)
        self._bs_list = []

    def init(self, warmup=False):
        self.model = Model(backend=self.args.backend, model_path=self.args.cls_model_path,
                           device_id=self.args.device_id)
        shape_type, shape_info = self.model.get_shape_info()

        if shape_type not in (ShapeType.DYNAMIC_BATCHSIZE, ShapeType.STATIC_SHAPE):
            raise ValueError("Input shape must be static shape or dynamic batch_size for classification model.")

        if shape_type == ShapeType.DYNAMIC_BATCHSIZE:
            self._bs_list, _, model_height, model_width = shape_info
        else:
            batchsize, _, model_height, model_width = shape_info
            self._bs_list = [batchsize]

        preprocess_ops = build_preprocess(self.args.cls_config_path)
        self.preprocess_ops = functools.partial(preprocess_ops, image_shape=(model_height, model_width))
        self.postprocess_ops = build_postprocess(self.args.cls_config_path)

        if warmup:
            self.model.warmup()

    def __call__(self, image: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        inputs = [image] if isinstance(image, np.ndarray) else image
        split_inputs_bs, split_outputs = self.preprocess(inputs)
        split_outputs = [self.model_infer(output["image"]) for output in split_outputs]

        outputs = []
        for bacth, split_output in zip(split_inputs_bs, split_outputs):
            output = split_output[:bacth, ...]
            output = self.postprocess_ops(output)
            outputs.append(output)

        return outputs[0] if isinstance(image, np.ndarray) else outputs

    def preprocess(self, image: List[np.ndarray]) -> Tuple[List[int], List[np.ndarray]]:
        num_image = len(image)
        batch_list = gear_utils.get_matched_gear_bs(len(image), self._bs_list)
        start_index = 0
        split_inputs_bs = []
        split_outputs = []
        for batch in batch_list:
            upper_bound = min(start_index + batch, num_image)
            split_input = image[start_index:upper_bound]
            split_output = self.preprocess_ops(split_input)
            split_output = gear_utils.padding_to_batch(split_output, batch)
            split_inputs_bs.append(upper_bound - start_index)
            split_outputs.append(split_output)
            start_index += batch

        return split_inputs_bs, split_outputs

    def model_infer(self, input: np.ndarray) -> np.ndarray:
        return self.model.infer([input])

    def postprocess(self, input: np.ndarray) -> List:
        return self.postprocess_ops(input)
