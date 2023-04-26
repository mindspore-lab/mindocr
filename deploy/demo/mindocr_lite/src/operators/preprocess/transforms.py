import math
from typing import Union, List

import cv2
import numpy as np

from ..utils import constant
from ...utils import to_chw_image, expand, get_hw_of_img, safe_div


class RGB2BGR:
    def __call__(self, image: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(image, (list, tuple)):
            dst_image = [self.rgb2bgr(img) for img in image]
        else:
            dst_image = self.rgb2bgr(image)

        return dst_image

    @staticmethod
    def rgb2bgr(image_src: np.ndarray):
        img = cv2.cvtColor(image_src, cv2.COLOR_RGB2BGR)
        return img


class NormalizeImage:
    def __init__(self,
                 std=constant.NORMALIZE_STD,
                 mean=constant.NORMALIZE_MEAN):
        self.std = np.array(std).reshape((1, 1, 3)).astype(np.float32)
        self.mean = np.array(mean).reshape((1, 1, 3)).astype(np.float32)

    def __call__(self, image: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(image, (list, tuple)):
            dst_image = [self.normalize(img, self.std, self.mean) for img in image]
        else:
            dst_image = self.normalize(image, self.std, self.mean)

        return dst_image

    @staticmethod
    def normalize(image_src: np.ndarray, std, mean):
        image_dst = image_src.astype(np.float32)
        image_dst = safe_div((image_dst - mean), std)
        return image_dst


class ToNCHW:
    def __call__(self, image: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        dst_image = image if isinstance(image, (list, tuple)) else [image]
        dst_image = [to_chw_image(img) for img in dst_image]
        dst_image = expand(dst_image)
        return dst_image


class LimitMaxSide:
    def __init__(self, limit_side=constant.LIMIT_SIDE):
        self.limit_side = limit_side

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.resize_by_limit_max_side(image, self.limit_side)

    @staticmethod
    def resize_by_limit_max_side(src_image: np.ndarray, limit_side: int):
        ratio = 1
        height, width = get_hw_of_img(src_image)
        if max(height, width) > limit_side:
            if height > width:
                ratio = safe_div(limit_side, height)
            else:
                ratio = safe_div(limit_side, width)

        resize_h = int(height * ratio)
        resize_w = int(width * ratio)

        resize_h = max(int(round(safe_div(resize_h, 32)) * 32), 32)
        resize_w = max(int(round(safe_div(resize_w, 32)) * 32), 32)
        dst_image = cv2.resize(src_image, (int(resize_w), int(resize_h)))
        return dst_image


class ResizeKeepAspectRatio:
    def __call__(self,
                 image: Union[np.ndarray, List[np.ndarray]],
                 dst_hw: tuple) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(image, (list, tuple)):
            dst_image = [self.resize(img, dst_hw) for img in image]
        else:
            dst_image = self.resize(image, dst_hw)

        return dst_image

    @staticmethod
    def resize(image, dst_hw):
        # just consider height and keep width
        dst_h, dst_w = dst_hw
        h, w = get_hw_of_img(image)
        ratio = safe_div(w, h)
        if math.ceil(ratio * dst_h) > dst_w:
            resize_w = dst_w
        else:
            resize_w = math.ceil(dst_h * ratio)

        dst_image = cv2.resize(image, (resize_w, dst_h))
        dst_image = ResizeKeepAspectRatio.padding_with_cv(dst_image, (dst_h, dst_w))

        return dst_image

    @staticmethod
    def padding_with_cv(image_src: np.array, gear: tuple):
        height, width = get_hw_of_img(image_src=image_src)
        gear_h, gear_w = gear
        padding_h, padding_w = gear_h - height, gear_w - width
        image_dst = cv2.copyMakeBorder(image_src, 0, padding_h, 0, padding_w, cv2.BORDER_CONSTANT, value=0.)
        return image_dst
    
    
class Resize:
    def __call__(self,
                 image: Union[np.ndarray, List[np.ndarray]],
                 dst_hw: tuple) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(image, (list, tuple)):
            dst_image = [self.resize(img, dst_hw) for img in image]
        else:
            dst_image = self.resize(image, dst_hw)

        return dst_image

    @staticmethod
    def resize(image, dst_hw):
        dst_h, dst_w = dst_hw
        dst_image = cv2.resize(image, (dst_w, dst_h))

        return dst_image

