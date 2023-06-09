import random
from typing import List, Union

import cv2
import numpy as np
from PIL import Image

from mindspore.dataset.vision import RandomColorAdjust as MSRandomColorAdjust
from mindspore.dataset.vision import ToPIL

from ...data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

__all__ = [
    "DecodeImage",
    "NormalizeImage",
    "ToCHWImage",
    "PackLoaderInputs",
    "RandomScale",
    "RandomColorAdjust",
]


class DecodeImage:
    """
    img_mode (str): The channel order of the output, 'BGR' and 'RGB'. Default to 'BGR'.
    channel_first (bool): if True, image shpae is CHW. If False, HWC. Default to False
    """

    def __init__(
        self, img_mode="BGR", channel_first=False, to_float32=False, ignore_orientation=False, keep_ori=False, **kwargs
    ):
        self.img_mode = img_mode
        self.to_float32 = to_float32
        self.channel_first = channel_first
        self.flag = cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR if ignore_orientation else cv2.IMREAD_COLOR
        self.keep_ori = keep_ori

    def __call__(self, data):
        if "img_path" in data:
            with open(data["img_path"], "rb") as f:
                img = f.read()
        elif "img_lmdb" in data:
            img = data["img_lmdb"]
        img = np.frombuffer(img, dtype="uint8")
        img = cv2.imdecode(img, self.flag)

        if self.img_mode == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        if self.to_float32:
            img = img.astype("float32")
        data["image"] = img
        # data['ori_image'] = img.copy()
        data["raw_img_shape"] = img.shape[:2]

        if self.keep_ori:
            data["image_ori"] = img.copy()

        return data


class NormalizeImage:
    """
    normalize image, substract mean, divide std
    input image: by default, np.uint8, [0, 255], HWC format.
    return image: float32 numpy array
    """

    def __init__(
        self,
        mean: Union[List[float], str] = "imagenet",
        std: Union[List[float], str] = "imagenet",
        is_hwc=True,
        bgr_to_rgb=False,
        rgb_to_bgr=False,
        **kwargs,
    ):
        # By default, imagnet MEAN and STD is in RGB order. inverse if input image is in BGR mode
        self._channel_conversion = False
        if bgr_to_rgb or rgb_to_bgr:
            self._channel_conversion = True

        # TODO: detect hwc or chw automatically
        shape = (3, 1, 1) if not is_hwc else (1, 1, 3)
        self.mean = np.array(self._get_value(mean, "mean")).reshape(shape).astype("float32")
        self.std = np.array(self._get_value(std, "std")).reshape(shape).astype("float32")
        self.is_hwc = is_hwc

    def __call__(self, data):
        img = data["image"]
        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"

        if self._channel_conversion:
            if self.is_hwc:
                img = img[..., [2, 1, 0]]
            else:
                img = img[[2, 1, 0], ...]

        data["image"] = (img.astype("float32") - self.mean) / self.std
        return data

    @staticmethod
    def _get_value(val, name):
        if isinstance(val, str) and val.lower() == "imagenet":
            assert name in ["mean", "std"]
            return IMAGENET_DEFAULT_MEAN if name == "mean" else IMAGENET_DEFAULT_STD
        elif isinstance(val, list):
            return val
        else:
            raise ValueError(f"Wrong {name} value: {val}")


class ToCHWImage:
    # convert hwc image to chw image
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data["image"]
        if isinstance(img, Image.Image):
            img = np.array(img)
        data["image"] = img.transpose((2, 0, 1))
        return data


class PackLoaderInputs:
    """
    Args:
        output_columns (list): the keys in data dict that are expected to output for dataloader

    Call:
        input: data dict
        output: data tuple corresponding to the `output_columns`
    """

    def __init__(self, output_columns: List, **kwargs):
        self.output_columns = output_columns

    def __call__(self, data):
        out = []
        for k in self.output_columns:
            assert k in data, f"key {k} does not exists in data, availabe keys are {data.keys()}"
            out.append(data[k])

        return tuple(out)


class RandomScale:
    """
    Randomly scales an image and its polygons in a predefined scale range.
    Args:
        scale_range: (min, max) scale range.
        p: probability of the augmentation being applied to an image.
    """

    def __init__(self, scale_range: Union[tuple, list], p: float = 0.5, **kwargs):
        self._range = scale_range
        self._p = p
        assert kwargs.get("is_train", True), ValueError("RandomScale augmentation must be used for training only")

    def __call__(self, data: dict):
        """
        required keys:
            image, HWC
            (polys)
        modified keys:
            image
            (polys)
        """
        if random.random() < self._p:
            scale = np.random.uniform(*self._range)
            data["image"] = cv2.resize(data["image"], dsize=None, fx=scale, fy=scale)
            if "polys" in data:
                data["polys"] *= scale

        return data


class RandomColorAdjust:
    def __init__(self, brightness=32.0 / 255, saturation=0.5, **kwargs):
        contrast = kwargs.get("contrast", (1, 1))
        hue = kwargs.get("hue", (0, 0))
        self._jitter = MSRandomColorAdjust(brightness=brightness, saturation=saturation, contrast=contrast, hue=hue)
        self._pil = ToPIL()

    def __call__(self, data):
        """
        required keys: image
        modified keys: image
        """
        # there's a bug in MindSpore that requires images to be converted to the PIL format first
        data["image"] = np.array(self._jitter(self._pil(data["image"])))
        return data
