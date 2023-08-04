import functools
from typing import Dict, List, Tuple, Union

import numpy as np

from ..core import Model, ShapeType
from ..data_process import build_postprocess, build_preprocess, gear_utils
from .infer_base import InferBase


class TextClassifier(InferBase):
    def __init__(self, args):
        super(TextClassifier, self).__init__(args)

    def _init_preprocess(self):
        preprocess_ops = build_preprocess(self.args.cls_config_path)
        self.preprocess_ops = functools.partial(preprocess_ops, target_size=self._hw_list[0])

    def _init_model(self):
        self.model = Model(
            backend=self.args.backend,
            device=self.args.device,
            model_path=self.args.cls_model_path,
            device_id=self.args.device_id,
        )

        shape_type, shape_value = self.model.get_shape_details()

        if shape_type not in (ShapeType.DYNAMIC_BATCHSIZE, ShapeType.STATIC_SHAPE):
            raise ValueError("Input shape must be static shape or dynamic batch_size for classification model.")

        # Only support NCHW format for net_inputs[0].
        # If multi input, the target_size may be invalid.
        shape_value = shape_value[0]

        if shape_type == ShapeType.DYNAMIC_BATCHSIZE:
            self._bs_list, _, h, w = shape_value
        else:
            batchsize, _, h, w = shape_value
            self._bs_list = [batchsize]

        self._hw_list = [(h, w)]

    def _init_postprocess(self):
        self.postprocess_ops = build_postprocess(self.args.cls_config_path)

    def get_params(self):
        return {"cls_batch_num": self._bs_list}

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
        return self.model.infer(data["net_inputs"])

    def postprocess(self, pred: List[np.ndarray], batch=None) -> Dict[str, List]:
        pred = gear_utils.get_batch_from_padding(pred, batch)
        pred = pred[0] if len(pred) == 1 else tuple(pred)
        results = self.postprocess_ops(pred)  # {"angles": angles, "scores": scores}
        return results
