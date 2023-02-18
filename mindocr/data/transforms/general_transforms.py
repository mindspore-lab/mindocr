from __future__ import absolute_import
from __future__ import division

from typing import List
import cv2
import math
import pyclipper
from shapely.geometry import Polygon
import numpy as np
from PIL import Image

from mindcv.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# IMAGENET_DEFAULT_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
# IMAGENET_DEFAULT_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]

__all__ = ['DecodeImage', 'NormalizeImage', 'ToCHWImage', 'PackLoaderInputs']

class DecodeImage(object):
    '''
    img_mode (str): The channel order of the output, 'BGR' and 'RGB'. Default to 'BGR'.
    channel_first (bool): if True, image shpae is CHW. If False, HWC. Default to False

    '''
    def __init__(self, img_mode='BGR', channel_first=False, to_float32=False, ignore_orientation=False, **kwargs):
        self.img_mode = img_mode
        self.to_float32 = to_float32
        self.channel_first = channel_first
        self.flag = cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR if ignore_orientation else cv2.IMREAD_COLOR

    def __call__(self, data):
        # TODO: use more efficient image loader, read binary, then decode?
        # TODO: why float32 in modelzoo. uint8 is more efficient

        #img = cv2.imread(data['img_path'], self.flag)
        # read from buffer is faster?
        with open(data['img_path'], 'rb') as f:
                img = f.read()
        img = np.frombuffer(img, dtype='uint8')
        img = cv2.imdecode(img, self.flag)

        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        if self.to_float32:
            img = img.astype('float32')
        data['image'] = img
        #data['ori_image'] = img.copy()
        return data


class NormalizeImage(object):
    """ normalize image, substract mean, divide std
    input image: by default, np.uint8, [0, 255], HWC format.
    return image: float32 numpy array
    """

    def __init__(self, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, is_hwc=True, bgr_to_rgb=False, rgb_to_bgr=False, **kwargs):
        # By default, imagnet MEAN and STD is in RGB order. inverse if input image is in BGR mode
        self._channel_conversion = False
        if bgr_to_rgb or rgb_to_bgr:
            self._channel_conversion = True

        # TODO: detect hwc or chw automatically
        shape = (3, 1, 1) if not is_hwc else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')
        self.is_hwc = is_hwc

    def __call__(self, data):
        img = data['image']
        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"

        if self._channel_conversion:
            if self.is_hwc:
                img = img[..., [2,1,0]]
            else:
                img = img[[2,1,0], ...]

        data['image'] = (
            img.astype('float32') - self.mean) / self.std
        return data

class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data['image']
        if isinstance(img, Image.Image):
            img = np.array(img)
        data['image'] = img.transpose((2, 0, 1))
        return data

class PackLoaderInputs(object):
    '''
    Args:
        output_keys (list): the keys in data dict that are expected to output for dataloader

    Call:
        input: data dict
        output: data tuple corresponding to the `output_keys`
    '''
    def __init__(self, output_keys: List, **kwargs):
        self.output_keys = output_keys

    def __call__(self, data):
        # TOOD: add assert for inexisted keys
        out = []
        for k in self.output_keys:
            assert k in data, f'key {k} does not exists in data, availabe keys are {data.keys()}'
            out.append(data[k])
        #data_tuple = tuple(data[k] for k in self.output_keys if )

        return tuple(out)
