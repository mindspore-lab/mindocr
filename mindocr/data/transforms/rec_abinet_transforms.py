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

from .svtr_transform import (
    CVColorJitter,
    CVGaussianNoise,
    CVMotionBlur,
    CVRandomAffine,
    CVRandomPerspective,
    CVRandomRotation,
    CVRescale,
)

__all__ = ["ABINetTransforms", "ABINetRecAug", "ABINetEval", "ABINetEvalTransforms"]


class ABINetTransforms(object):
    """Convert text label (str) to a sequence of character indices according to the char dictionary

    Args:

    """

    def __init__(
        self,
    ):
        # ABINet_Transforms
        self.case_sensitive = False
        self.charset = CharsetMapper(max_length=26)

    def __call__(self, data: dict):
        img_lmdb = data["img_lmdb"]
        label = data["label"]
        label = label.encode("utf-8")
        label = str(label, "utf-8")
        try:
            label = re.sub("[^0-9a-zA-Z]+", "", label)
            if len(label) > 25 or len(label) <= 0:
                string_false2 = f"en(label) > 25 or len(label) <= 0:   {label}, {len(label)}"
                logging.info(string_false2)
            label = label[:25]
            buf = six.BytesIO()
            buf.write(img_lmdb)
            buf.seek(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                image = PIL.Image.open(buf).convert("RGB")
            if not _check_image(image, pixels=6):
                string_false1 = f"_check_image false:   {label}, {len(label)}"
                logging.info(string_false1)
        except Exception:
            string_false = f"Corrupted image is found:   {label}, {len(label)}"
            logging.info(string_false)

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
    def __init__(self):
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
        self.w = 128
        self.h = 32

    def __call__(self, data):
        img = data["image"]
        img = self.transforms(img)
        img = cv2.resize(img, (self.w, self.h))
        img = self.toTensor(img)
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
        # start_token='<BOS>',
        # end_token='<EOS>',
        # unkown_token='',
    ):
        # ABINet_Transforms
        self.case_sensitive = False
        self.charset = CharsetMapper(max_length=26)

    def __call__(self, data: dict):
        img_lmdb = data["img_lmdb"]
        label = data["label"]
        label = label.encode("utf-8")
        label = str(label, "utf-8")
        try:
            label = re.sub("[^0-9a-zA-Z]+", "", label)
            if len(label) > 25 or len(label) <= 0:
                string_false2 = f"en(label) > 25 or len(label) <= 0:   {label}, {len(label)}"
                logging.info(string_false2)
            label = label[:25]
            buf = six.BytesIO()
            buf.write(img_lmdb)
            buf.seek(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                image = PIL.Image.open(buf).convert("RGB")
            if not _check_image(image, pixels=6):
                string_false1 = f"_check_image false:   {label}, {len(label)}"
                logging.info(string_false1)
        except Exception:
            string_false = f"Corrupted image is found:   {label}, {len(label)}"
            logging.info(string_false)

        image = np.array(image)

        text = label

        length = len(text) + 1
        length = float(length)

        label = self.charset.get_labels(text, case_sensitive=self.case_sensitive)
        label_for_mask = copy.deepcopy(label)
        label_for_mask[int(length - 1)] = 1
        label = onehot(label, self.charset.num_classes)
        data_dict = {"image": image, "label": text, "length": length, "label_for_mask": label_for_mask}
        return data_dict


class ABINetEval(object):
    def __init__(self):
        self.toTensor = ds.vision.ToTensor()
        self.w = 128
        self.h = 32

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
    def __init__(self, degrees=15, translate=(0.3, 0.3), scale=(0.5, 2.0), shear=(45, 15), distortion=0.5, p=0.5):
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
    def __init__(self, var, degrees, factor, p=0.5):
        self.p = p
        transforms = []
        if var is not None:
            transforms.append(CVGaussianNoise(var=var))
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


class CharsetMapper(object):
    def __init__(self, max_length=30, null_char="\u2591"):
        self.null_char = null_char
        self.max_length = max_length
        self.label_to_char = self._read_charset()
        self.char_to_label = dict(map(reversed, self.label_to_char.items()))
        self.num_classes = len(self.label_to_char)

    def _read_charset(self):
        charset = {}
        charset = {
            0: "░",
            1: "a",
            2: "b",
            3: "c",
            4: "d",
            5: "e",
            6: "f",
            7: "g",
            8: "h",
            9: "i",
            10: "j",
            11: "k",
            12: "l",
            13: "m",
            14: "n",
            15: "o",
            16: "p",
            17: "q",
            18: "r",
            19: "s",
            20: "t",
            21: "u",
            22: "v",
            23: "w",
            24: "x",
            25: "y",
            26: "z",
            27: "1",
            28: "2",
            29: "3",
            30: "4",
            31: "5",
            32: "6",
            33: "7",
            34: "8",
            35: "9",
            36: "0",
        }
        self.null_label = 0
        charset[self.null_label] = self.null_char
        return charset

    def trim(self, text):
        assert isinstance(text, str)
        return text.replace(self.null_char, "")

    def get_text(self, labels, length=None, padding=True, trim=False):
        """Returns a string corresponding to a sequence of character ids."""
        length = length if length else self.max_length
        labels = [int(a) if isinstance(a, ms.Tensor) else int(a) for a in labels]
        if padding:
            labels = labels + [self.null_label] * (length - len(labels))
        text = "".join([self.label_to_char[label] for label in labels])
        if trim:
            text = self.trim(text)
        return text

    def get_labels(self, text, length=None, padding=True, case_sensitive=False):
        """Returns the labels of the corresponding text."""
        length = length if length else self.max_length
        if padding:
            text = text + self.null_char * (length - len(text))
        if not case_sensitive:
            text = text.lower()
        labels = [self.char_to_label[char] for char in text]
        return labels

    def pad_labels(self, labels, length=None):
        length = length if length else self.max_length
        return labels + [self.null_label] * (length - len(labels))

    @property
    def digits(self):
        return "0123456789"

    @property
    def digit_labels(self):
        return self.get_labels(self.digits, padding=False)

    @property
    def alphabets(self):
        all_chars = list(self.char_to_label.keys())
        valid_chars = []
        for c in all_chars:
            if c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
                valid_chars.append(c)
        return "".join(valid_chars)

    @property
    def alphabet_labels(self):
        return self.get_labels(self.alphabets, padding=False)


def onehot(label, depth, device=None):
    label_shape = 26

    onehot_output = np.zeros((label_shape, depth))

    label_expand = np.expand_dims(label, -1)
    label_expand = np.expand_dims(label_expand, -1)
    label_expand_onehot = np.zeros((26, 37))
    a = 0
    for i in label_expand:
        i = int(i)
        label_expand_onehot[a][i] = 1
        a = a + 1

    label_expand_onehot = label_expand_onehot
    onehot_output = label_expand_onehot + onehot_output

    return onehot_output
