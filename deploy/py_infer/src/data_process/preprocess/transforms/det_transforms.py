import os
import sys

import cv2

# add mindocr root path, and import transforms from mindocr
mindocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
sys.path.insert(0, mindocr_path)

from mindocr.data.transforms import det_transforms

__all__ = ["DetResize", "ScalePadImage"]


class DetResize(det_transforms.DetResize):
    # limit_type and force_divisable is not supported currently, because inference model don't support dynamic shape
    def __init__(self,
                 keep_ratio=True,
                 padding=True,
                 interpolation=cv2.INTER_LINEAR,
                 **kwargs):
        if keep_ratio and (not padding):
            print(f"WARNING: output shape can be dynamic if keep_ratio but no padding, "
                  f"but inference don't support dynamic shape, so padding is reset to True.")
            padding = True

        skipped = ("target_size", "limit_type", "force_divisable")
        [kwargs.pop(name, None) for name in skipped]

        super().__init__(target_size=None, keep_ratio=keep_ratio, padding=padding, limit_type=None,
                         force_divisable=False, interpolation=interpolation, **kwargs)

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
