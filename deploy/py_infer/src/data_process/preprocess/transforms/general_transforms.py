from collections import defaultdict
from typing import List

import cv2
import numpy as np

__all__ = ["DecodeImage", "ToBatch"]


class DecodeImage:
    def __init__(self, img_mode="RGB", channel_first=False, to_float32=False, **kwargs):
        self.img_mode = img_mode
        self.to_float32 = to_float32
        self.channel_first = channel_first

    def __call__(self, data):
        data["image"] = self._decode(data["image"])
        return data

    def _decode(self, img):
        if self.img_mode == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        if self.to_float32:
            img = img.astype("float32")

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
