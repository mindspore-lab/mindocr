import cv2
import numpy as np

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))

from mindocr.data.layout_dataset import xyxy2xywh

def letterbox(scaleup):
    def func(data):
        image = data["image"]
        hw_ori = data["raw_img_shape"]
        new_shape = data["target_size"]
        color = (114, 114, 114)
        # Resize and pad image while meeting stride-multiple constraints
        shape = image.shape[:2]  # current shape [height, width]
        h, w = shape[:]
        # h0, w0 = hw_ori
        h0, w0 = new_shape
        # hw_scale = np.array([h / h0, w / w0])
        hw_scale = np.array([h0 / h, w0 / w])
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw, dh = dw / 2, dh / 2  # divide padding into 2 sides
        hw_pad = np.array([dh, dw])

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        data["image"] = image
        data["image_ids"] = 0
        data["hw_ori"] = hw_ori
        data["hw_scale"] = hw_scale
        data["pad"] = hw_pad
        return data

    return func


def image_norm(scale=255.0):
    def func(data):
        image = data["image"]
        image = image.astype(np.float32, copy=False)
        image /= scale
        data["image"] = image
        return data

    return func


def image_transpose(bgr2rgb=True, hwc2chw=True):
    def func(data):
        image = data["image"]
        if bgr2rgb:
            image = image[:, :, ::-1]
        if hwc2chw:
            image = image.transpose(2, 0, 1)
        data["image"] = image
        return data

    return func

def label_norm(labels, xyxy2xywh_=True):
    def func(data):
        if len(labels) == 0:
            return data, labels

        if xyxy2xywh_:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh

        labels[:, [2, 4]] /= data.shape[0]  # normalized height 0-1
        labels[:, [1, 3]] /= data.shape[1]  # normalized width 0-1

        return data, labels
    return func