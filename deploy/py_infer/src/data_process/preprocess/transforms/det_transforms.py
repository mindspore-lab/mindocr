import math
import os
import sys

import cv2
import numpy as np

# add mindocr root path, and import transforms from mindocr
mindocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
sys.path.insert(0, mindocr_path)

from mindocr.data.transforms import det_transforms  # noqa

__all__ = ["DetResize", "ScalePadImage", "DetResizeNormForInfer"]


class DetResize(det_transforms.DetResize):
    # limit_type and force_divisable is not supported currently, because inference model don't support dynamic shape
    def __init__(self, keep_ratio=True, padding=True, interpolation=cv2.INTER_LINEAR, **kwargs):
        if keep_ratio and (not padding):
            print(
                "WARNING: output shape can be dynamic if keep_ratio but no padding for DetResize, "
                "but inference doesn't support dynamic shape, so padding is reset to True."
            )
            padding = True

        skipped = ("target_size", "limit_type", "force_divisable")
        [kwargs.pop(name, None) for name in skipped]

        super().__init__(
            target_size=None,
            keep_ratio=keep_ratio,
            padding=padding,
            limit_type=None,
            force_divisable=False,
            interpolation=interpolation,
            **kwargs,
        )

    # move 'target_size' to __call__ from __init__
    def __call__(self, data):
        self.target_size = data["target_size"]
        return super().__call__(data)


class ScalePadImage(det_transforms.ScalePadImage):
    def __init__(self, **kwargs):
        skipped = ("target_size",)
        [kwargs.pop(name, None) for name in skipped]

        super().__init__(target_size=None, **kwargs)

    # move 'target_size' to __call__ from __init__
    def __call__(self, data: dict):
        self.target_size = data["target_size"]
        return super().__call__(data)


class DetResizeNormForInfer(object):
    def __init__(
        self,
        keep_ratio=True,
        padding=True,
        interpolation=cv2.INTER_LINEAR,
        norm_before_pad=False,
        mean=[127.0, 127.0, 127.0],
        std=[127.0, 127.0, 127.0],
        **kwargs
    ):
        if keep_ratio and (not padding):
            print(
                "WARNING: output shape can be dynamic if keep_ratio but no padding for DetResizeNormForInfer, "
                "but inference don't support dynamic shape, so padding is reset to True."
            )
            padding = True

        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.norm_before_pad = norm_before_pad
        self.mean = np.array(mean, dtype="float32")
        self.std = np.array(std, dtype="float32")

    def norm(self, img):
        return (img.astype(np.float32) - self.mean) / self.std

    def __call__(self, data):
        img = data["image"]

        h, w = img.shape[:2]
        tar_h, tar_w = data["target_size"]

        scale_ratio = 1.0

        if self.keep_ratio:
            scale_ratio = min(tar_h / h, tar_w / w)

        if self.keep_ratio:
            resize_w = min(math.ceil(w * scale_ratio), tar_w)
            resize_h = min(math.ceil(h * scale_ratio), tar_h)
        else:
            resize_w = tar_w
            resize_h = tar_h

        resized_img = cv2.resize(img, (resize_w, resize_h), interpolation=self.interpolation)

        if self.norm_before_pad:
            resized_img = self.norm(resized_img)

        if self.keep_ratio and (tar_h >= resize_h and tar_w >= resize_w):
            img = np.zeros((tar_h, tar_w, 3), dtype=resized_img.dtype)
            img[:resize_h, :resize_w, :] = resized_img
        else:
            img = resized_img

        if not self.norm_before_pad:
            img = self.norm(img)

        data["image"] = img

        scale_h = resize_h / h
        scale_w = resize_w / w

        if "shape_list" not in data:
            src_h, src_w = data.get("raw_img_shape", (h, w))
            data["shape_list"] = [src_h, src_w, scale_h, scale_w]
        else:
            data["shape_list"][2] = data["shape_list"][2] * scale_h
            data["shape_list"][3] = data["shape_list"][3] * scale_h

        return data
