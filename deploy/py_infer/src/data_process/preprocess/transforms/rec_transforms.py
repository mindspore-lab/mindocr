import math
import os
import sys

import cv2
import numpy as np

# add mindocr root path, and import transforms from mindocr
mindocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
sys.path.insert(0, mindocr_path)

from mindocr.data.transforms import rec_transforms  # noqa

__all__ = [
    "SVTRRecResizeImg",
    "RecResizeNormForInfer",
    "RecResizeNormForViTSTR",
    "RecResizeNormForMMOCR",
]


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


class RecResizeNormForMMOCR:
    def __init__(
        self,
        height=32,
        min_width=32,
        max_width=160,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        pad_width=160,
    ):
        self.height = height
        self.min_width = min_width
        self.max_width = max_width
        self.mean = mean
        self.std = std
        self.pad_width = pad_width

    def __call__(self, data):
        img = data["image"]
        ori_height, ori_width = img.shape[:2]
        new_width = math.ceil(float(self.height) / ori_height * ori_width)
        if "target_size" in data:
            self.tar_h, self.tar_w = data["target_size"]
            padded_img = np.zeros((self.tar_h, self.tar_w, 3), dtype=np.float32)
        else:
            padded_img = np.zeros((ori_height, self.pad_width, 3), dtype=np.float32)
        if self.min_width is not None:
            new_width = max(self.min_width, new_width)
        if self.max_width is not None:
            new_width = min(self.max_width, new_width)
        if self.tar_w is not None:
            new_width = min(self.tar_w, new_width)
        resized_img = cv2.resize(img, (new_width, self.height), interpolation=cv2.INTER_LINEAR)
        padded_img[:, :new_width, :] = resized_img
        padded_img = (padded_img - self.mean) / self.std
        data["image"] = padded_img.astype(np.float32)
        return data
