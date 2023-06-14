import os
import sys

import cv2
import numpy as np

# add mindocr root path, and import transforms from mindocr
mindocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
sys.path.insert(0, mindocr_path)

from mindocr.data.transforms import rec_transforms  # noqa

__all__ = ["SVTRRecResizeImg", "RecResizeNormForInfer", "RecResizeNormForViTSTR"]


class SVTRRecResizeImg(rec_transforms.SVTRRecResizeImg):
    def __init__(self, padding=True, **kwargs):
        skipped = ("image_shape",)
        [kwargs.pop(name, None) for name in skipped]

        super().__init__(image_shape=None, padding=padding, **kwargs)

    # move 'image_shape' to __call__ from __init__
    def __call__(self, data: dict):
        self.image_shape = data["target_size"]
        return super().__call__(data)


class RecResizeNormForInfer(rec_transforms.RecResizeNormForInfer):
    def __init__(
        self,
        keep_ratio=True,
        padding=False,
        interpolation=cv2.INTER_LINEAR,
        norm_before_pad=False,
        mean=[127.0, 127.0, 127.0],
        std=[127.0, 127.0, 127.0],
        **kwargs
    ):
        skipped = ("target_height", "target_width")
        [kwargs.pop(name, None) for name in skipped]

        super().__init__(
            target_height=None,
            target_width=None,
            keep_ratio=keep_ratio,
            padding=padding,
            interpolation=interpolation,
            norm_before_pad=norm_before_pad,
            mean=mean,
            std=std,
            **kwargs
        )

    # move 'target_height' and 'target_width' to __call__ from __init__
    def __call__(self, data):
        self.tar_h, self.tar_w = data["target_size"]
        return super().__call__(data)


class RecResizeNormForViTSTR(object):
    def __call__(self, data):
        self.tar_h, self.tar_w = data["target_size"]
        img = data["image"]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, [self.tar_w, self.tar_h], interpolation=cv2.INTER_CUBIC)
        img = np.array(img)
        norm_img = np.expand_dims(img, -1)
        norm_img = norm_img.astype(np.float32) / 255.0
        data["image"] = norm_img
        return data
