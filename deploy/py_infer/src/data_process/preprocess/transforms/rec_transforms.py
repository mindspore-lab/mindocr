import os
import sys

import cv2

# add mindocr root path, and import transforms from mindocr
mindocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
sys.path.insert(0, mindocr_path)

from mindocr.data.transforms import rec_transforms

__all__ = ["RecResizeImg", "SVTRRecResizeImg", "RecResizeNormForInfer"]


class RecResizeImg(rec_transforms.RecResizeImg):
    def __init__(self, padding=True, **kwargs):
        skipped = ("image_shape",)
        [kwargs.pop(name, None) for name in skipped]

        super().__init__(image_shape=None, padding=padding, **kwargs)

    # move 'image_shape' to __call__ from __init__
    def __call__(self, data: dict):
        self.image_shape = data["target_size"]
        return super().__call__(data)


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
    def __init__(self,
                 keep_ratio=True,
                 padding=False,
                 interpolation=cv2.INTER_LINEAR,
                 norm_before_pad=False,
                 mean=[127.0, 127.0, 127.0],
                 std=[127.0, 127.0, 127.0],
                 **kwargs):
        skipped = ("target_height", "target_width")
        [kwargs.pop(name, None) for name in skipped]

        super().__init__(target_height=None, target_width=None, keep_ratio=keep_ratio, padding=padding,
                         interpolation=interpolation, norm_before_pad=norm_before_pad, mean=mean, std=std, **kwargs)

    # move 'target_height' and 'target_width' to __call__ from __init__
    def __call__(self, data):
        self.tar_h, self.tar_w = data["target_size"]
        return super().__call__(data)
