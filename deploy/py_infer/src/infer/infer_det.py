from typing import Dict, List

import numpy as np

from ..core import Model, ShapeType
from ..data_process import build_postprocess, build_preprocess, cv_utils, gear_utils
from .infer_base import InferBase


class TextDetector(InferBase):
    def __init__(self, args):
        super(TextDetector, self).__init__(args)

    def _init_preprocess(self):
        self.preprocess_ops = build_preprocess(self.args.det_config_path)

    def _init_model(self):
        self.model = Model(
            backend=self.args.backend,
            model_path=self.args.det_model_path,
            device_id=self.args.device_id,
        )

        shape_type, shape_info = self.model.get_shape_info()

        if shape_type not in (ShapeType.DYNAMIC_IMAGESIZE, ShapeType.STATIC_SHAPE):
            raise ValueError("Input shape must be static shape or dynamic image_size for detection model.")

        if shape_type == ShapeType.DYNAMIC_IMAGESIZE:
            batchsize, _, hw_list = shape_info
        else:
            batchsize, _, h, w = shape_info
            hw_list = [(h, w)]

        if batchsize != 1:
            raise ValueError("Input batch size must be 1 for detection model.")

        self._hw_list = hw_list
        self._bs_list = [batchsize]

    def _init_postprocess(self):
        self.postprocess_ops = build_postprocess(self.args.det_config_path, rescale_fields=["polys"])

    def get_params(self):
        return {"det_batch_num": self._bs_list}

    def __call__(self, image: np.ndarray):
        data = self.preprocess(image)
        pred = self.model_infer(data)
        polys = self.postprocess(pred, data["shape_list"])

        return polys

    def preprocess(self, image: np.ndarray) -> Dict:
        target_size = gear_utils.get_matched_gear_hw(cv_utils.get_hw_of_img(image), self._hw_list)
        return self.preprocess_ops(image, target_size=target_size)

    def model_infer(self, data: Dict) -> List[np.ndarray]:
        return self.model.infer([data["image"]])  # model infer for single input

    def postprocess(self, pred, shape_list: np.ndarray) -> List[np.ndarray]:
        polys = self.postprocess_ops(tuple(pred), shape_list)["polys"][0]  # {'polys': [img0_polys, ...], ...}
        polys = [np.array(x) for x in polys]
        return polys  # [poly(points_num, 2), ...], bs=1
