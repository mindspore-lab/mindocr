"""
transform for text recognition tasks.
"""
import copy
import logging
import random
import re
import warnings

import cv2
import numpy as np
import PIL
import six
from PIL import Image

import mindspore as ms
import mindspore.dataset as ds

from ...models.utils.abinet_layers import CharsetMapper, onehot
from .svtr_transform import (
    CVColorJitter,
    CVGaussianNoise,
    CVMotionBlur,
    CVRandomAffine,
    CVRandomPerspective,
    CVRandomRotation,
    CVRescale,
)

_logger = logging.getLogger(__name__)
__all__ = ["ABINetTransforms", "ABINetRecAug", "ABINetEval", "ABINetEvalTransforms"]


class ABINetTransforms(object):
    """Convert text label (str) to a sequence of character indices according to the char dictionary

    Args:

    """

    def __init__(
        self,
        **kwargs,
    ):
        # ABINet_Transforms
        self.case_sensitive = False
        self.charset = CharsetMapper(max_length=26)

    def __call__(self, data: dict):
        if "img_path" in data:
            with open(data["img_path"], "rb") as f:
                img = f.read()
        elif "img_lmdb" in data:
            img = data["img_lmdb"]
        label = data["label"]
        label = label.encode("utf-8")
        label = str(label, "utf-8")
        try:
            label = re.sub("[^0-9a-zA-Z]+", "", label)
            if len(label) > 25 or len(label) <= 0:
                string_false2 = f"len(label) > 25 or len(label) <= 0:   {label}, {len(label)}"
                _logger.warning(string_false2)
            label = label[:25]
            buf = six.BytesIO()
            buf.write(img)
            buf.seek(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                image = PIL.Image.open(buf).convert("RGB")
            if not _check_image(image, pixels=6):
                string_false1 = f"_check_image false:   {label}, {len(label)}"
                _logger.warning(string_false1)
        except Exception:
            string_false = f"Corrupted image is found:   {label}, {len(label)}"
            _logger.warning(string_false)

        image = np.array(image)

        text = label

        length = len(text) + 1
        length = float(length)

        label = self.charset.get_labels(text, case_sensitive=self.case_sensitive)
        label_for_mask = copy.deepcopy(label)
        label_for_mask[int(length - 1)] = 1
        label = onehot(label, self.charset.num_classes)
        data_dict = {"image": image, "label": label, "length": length, "label_for_mask": label_for_mask}
        return data_dict


class ABINetRecAug(object):
    def __init__(self, width=128, height=32, **kwargs):
        self.transforms = ds.transforms.Compose(
            [
                CVGeometry(
                    degrees=45,
                    translate=(0.0, 0.0),
                    scale=(0.5, 2.0),
                    shear=(45, 15),
                    distortion=0.5,
                    p=0.5,
                ),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25),
            ]
        )
        self.toTensor = ds.vision.ToTensor()
        self.w = width
        self.h = height
        self.op = ms.dataset.vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)

    def __call__(self, data):
        img = data["image"]
        img = self.transforms(img)
        img = cv2.resize(img, (self.w, self.h))
        img = self.toTensor(img)
        img = self.op(img)
        data["image"] = img
        return data


def _check_image(x, pixels=6):
    if x.size[0] <= pixels or x.size[1] <= pixels:
        return False
    else:
        return True


class ABINetEvalTransforms(object):
    """Convert text label (str) to a sequence of character indices according to the char dictionary

    Args:

    """

    def __init__(
        self,
        **kwargs,
    ):
        # ABINet_Transforms
        self.case_sensitive = False
        self.charset = CharsetMapper(max_length=26)

    def __call__(self, data: dict):
        if "img_path" in data:
            with open(data["img_path"], "rb") as f:
                img = f.read()
        elif "img_lmdb" in data:
            img = data["img_lmdb"]
        label = data["label"]
        label = label.encode("utf-8")
        label = str(label, "utf-8")
        try:
            label = re.sub("[^0-9a-zA-Z]+", "", label)
            if len(label) > 25 or len(label) <= 0:
                string_false2 = f"en(label) > 25 or len(label) <= 0:   {label}, {len(label)}"
                _logger.warning(string_false2)
            label = label[:25]
            buf = six.BytesIO()
            buf.write(img)
            buf.seek(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                image = PIL.Image.open(buf).convert("RGB")
            if not _check_image(image, pixels=6):
                string_false1 = f"_check_image false:   {label}, {len(label)}"
                _logger.warning(string_false1)
        except Exception:
            string_false = f"Corrupted image is found:   {label}, {len(label)}"
            _logger.warning(string_false)

        image = np.array(image)

        text = label
        length = len(text) + 1
        length = float(length)
        data_dict = {"image": image, "label": text, "length": length}
        return data_dict


class ABINetEval(object):
    def __init__(self, **kwargs):
        self.toTensor = ds.vision.ToTensor()
        self.w = 128
        self.h = 32
        self.op = ms.dataset.vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)

    def __call__(self, data):
        img = data["image"]
        img = cv2.resize(img, (self.w, self.h))
        img = self.toTensor(img)
        data["image"] = img
        length = data["length"]
        length = int(length)
        data["length"] = length
        return data


class CVGeometry(object):
    def __init__(
        self, degrees=15, translate=(0.3, 0.3), scale=(0.5, 2.0), shear=(45, 15), distortion=0.5, p=0.5, **kwargs
    ):
        self.p = p
        type_p = random.random()
        if type_p < 0.33:
            self.transforms = CVRandomRotation(degrees=degrees)
        elif type_p < 0.66:
            self.transforms = CVRandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear)
        else:
            self.transforms = CVRandomPerspective(distortion=distortion)

    def __call__(self, img):
        if random.random() < self.p:
            img = np.array(img)
            return Image.fromarray(self.transforms(img))
        else:
            return img


class CVDeterioration(object):
    def __init__(self, var, degrees, factor, p=0.5, **kwargs):
        self.p = p
        transforms = []
        if var is not None:
            transforms.append(CVGaussianNoise(variance=var))
        if degrees is not None:
            transforms.append(CVMotionBlur(degrees=degrees))
        if factor is not None:
            transforms.append(CVRescale(factor=factor))

        random.shuffle(transforms)

        transforms = ds.transforms.Compose(transforms)
        self.transforms = transforms

    def __call__(self, img):
        if random.random() < self.p:
            img = np.array(img)
            return Image.fromarray(self.transforms(img))
        else:
            return img
