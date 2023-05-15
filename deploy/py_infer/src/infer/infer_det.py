from typing import Union

import numpy as np

from .infer_base import InferBase
from ..data_process import gear_utils, cv_utils, build_preprocess, build_postprocess
from ..core import Model, ShapeType


class TextDetector(InferBase):
    def __init__(self, args):
        super(TextDetector, self).__init__(args)
        self._hw_list = []

    def init(self, warmup=False):
        self.model = Model(backend=self.args.backend, model_path=self.args.det_model_path,
                           device_id=self.args.device_id)

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
        self.preprocess_ops = build_preprocess(self.args.det_config_path)
        self.postprocess_ops = build_postprocess(self.args.det_config_path)

        if warmup:
            self.model.warmup()

    def __call__(self, image: np.ndarray) -> Union[list, np.ndarray]:
        output = self.preprocess(image)
        pred = self.model_infer(output["image"])
        polys = self.postprocess(pred, output["shape"])

        return polys

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        dst_hw = gear_utils.get_matched_gear_hw(cv_utils.get_hw_of_img(image), self._hw_list)
        return self.preprocess_ops(image, image_shape=dst_hw)

    def model_infer(self, input: np.ndarray):
        return self.model.infer([input])

    def postprocess(self, input, shape) -> Union[list, np.ndarray]:
        ploys, *_ = self.postprocess_ops(input, shape)
        return ploys[0]
