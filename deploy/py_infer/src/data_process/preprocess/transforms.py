from collections import defaultdict
from typing import Union, List

import cv2
import math
import numpy as np

from ..utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

__all__ = ["DecodeImage", "NormalizeImage", "ResizeImage", "ScalePadImage", "RecResizeImg", "ClsResizeImg",
           "ToCHWImage", "ToBatch"]


class DecodeImage:
    def __init__(self, img_mode='BGR', channel_first=False, to_float32=False):
        self.img_mode = img_mode
        self.to_float32 = to_float32
        self.channel_first = channel_first

    def __call__(self, data):
        data["image"] = self._fake_decode(data["image"])
        return data

    def _fake_decode(self, img):
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        if self.to_float32:
            img = img.astype('float32')

        return img


class NormalizeImage:
    def __init__(self, mean: Union[List[float], str] = 'imagenet', std: Union[List[float], str] = 'imagenet',
                 is_hwc=True, bgr_to_rgb=False, rgb_to_bgr=False):
        # By default, imagnet MEAN and STD is in RGB order. inverse if input image is in BGR mode
        self._channel_conversion = False
        if bgr_to_rgb or rgb_to_bgr:
            self._channel_conversion = True

        # TODO: detect hwc or chw automatically
        shape = (3, 1, 1) if not is_hwc else (1, 1, 3)
        self.mean = np.array(self._get_value(mean, 'mean')).reshape(shape).astype('float32')
        self.std = np.array(self._get_value(std, 'std')).reshape(shape).astype('float32')
        self.is_hwc = is_hwc

    def __call__(self, data):
        data["image"] = self._normalize(data["image"])
        return data

    def _normalize(self, img):
        if self._channel_conversion:
            if self.is_hwc:
                img = img[..., [2, 1, 0]]
            else:
                img = img[[2, 1, 0], ...]

        img = (img.astype('float32') - self.mean) / self.std
        return img

    @staticmethod
    def _get_value(val, name):
        if isinstance(val, str) and val.lower() == 'imagenet':
            assert name in ['mean', 'std']
            return IMAGENET_DEFAULT_MEAN if name == 'mean' else IMAGENET_DEFAULT_STD
        elif isinstance(val, list):
            return val
        else:
            raise ValueError(f'Wrong {name} value: {val}')


class ResizeImage:
    def __init__(self, keep_ratio=False):
        self._keep_ratio = keep_ratio

    def __call__(self, data):
        if self._keep_ratio:
            return self._resize_keep_ratio(data)
        else:
            return self._resize(data)

    def _resize(self, data):
        size = np.array(data['image'].shape[:2])
        target_size = data["image_shape"]  # src_w, src_h
        scale = target_size / size         # ratio_w, ratio_h
        data['image'] = cv2.resize(data['image'], target_size[::-1])
        data['shape'] = np.concatenate((size, scale), dtype=np.float32)

        return data

    def _resize_keep_ratio(self, data):
        size = np.array(data['image'].shape[:2])
        target_size = data["image_shape"]

        scale = min(target_size / size)
        new_size = np.round(scale * size).astype(np.int32)

        data['image'] = cv2.resize(data['image'], new_size[::-1])
        data['image'] = np.pad(data['image'],
                               (*tuple((0, ts - ns) for ts, ns in zip(target_size, new_size)), (0, 0)))

        # [src_h, src_w, ratio_h, ratio_w]
        data['shape'] = np.concatenate((size, np.array([scale, scale])), dtype=np.float32)

        return data


class ScalePadImage(ResizeImage):
    def __init__(self):
        super(ScalePadImage, self).__init__(keep_ratio=True)


class RecResizeImg:
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, data):
        data["image"] = self._resize(data["image"], data["image_shape"])
        return data

    def _resize(self, img, image_shape):
        return self._resize_norm_img(img, image_shape)

    def _resize_norm_img(self,
                         img,
                         image_shape,
                         interpolation=cv2.INTER_LINEAR):

        imgH, imgW = image_shape
        if not self.padding:
            resized_image = cv2.resize(
                img, (imgW, imgH), interpolation=interpolation)
            return resized_image
        else:
            h, w, c = img.shape
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
            resized_image = cv2.resize(img, (resized_w, imgH))

            padding_im = np.zeros((imgH, imgW, c), dtype=img.dtype)
            padding_im[:, 0:resized_w, :] = resized_image

            return padding_im


class ClsResizeImg(RecResizeImg):
    pass


class ToCHWImage:
    def __call__(self, data):
        data["image"] = self._to_chw_image(data["image"])
        return data

    def _to_chw_image(self, img):
        return img.transpose((2, 0, 1))


class ToBatch:
    def __call__(self, data_list: List[dict]) -> dict:
        outputs_list = {"image", "shape"}
        batch_data = defaultdict(list)
        for data in data_list:
            for key, value in data.items():
                if key in outputs_list:
                    batch_data[key].append(value)

        outputs = {}
        for key, value in batch_data.items():
            outputs[key] = np.array(value)

        return outputs
