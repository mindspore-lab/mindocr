import os
import sys
from collections import defaultdict
from typing import List

import cv2
import numpy as np

# add mindocr root path, and import transforms from mindocr
mindocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
sys.path.insert(0, mindocr_path)

from mindocr.data.transforms import general_transforms

__all__ = ["DecodeImage", "NormalizeImage", "ToCHWImage", "ToBatch"]

NormalizeImage = general_transforms.NormalizeImage
ToCHWImage = general_transforms.ToCHWImage


class DecodeImage:
    def __init__(self, img_mode='BGR', channel_first=False, to_float32=False, **kwargs):
        self.img_mode = img_mode
        self.to_float32 = to_float32
        self.channel_first = channel_first

    def __call__(self, data):
        data["image"] = self._decode(data["image"])
        return data

    def _decode(self, img):
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        if self.to_float32:
            img = img.astype('float32')

        return img


class ToBatch:
    def __init__(self, output_columns):
        self.output_columns = output_columns

    def __call__(self, data_list: List[dict]) -> dict:
        batch_data = defaultdict(list)
        for data in data_list:
            for key, value in data.items():
                if key in self.output_columns:
                    batch_data[key].append(value)

        outputs = {}
        for key, value in batch_data.items():
            outputs[key] = np.array(value)

        return outputs
