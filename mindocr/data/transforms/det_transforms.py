"""
transforms for text detection tasks.
"""
import warnings
from typing import List
import math

import json
import cv2
import pyclipper
from shapely.geometry import Polygon
import numpy as np

__all__ = ['DetLabelEncode', 'BorderMap', 'ShrinkBinaryMap', 'DetResize', 'expand_poly']


class DetLabelEncode:
    def __init__(self, **kwargs):
        pass

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def expand_points_num(self, boxes):
        max_points_num = 0
        for box in boxes:
            if len(box) > max_points_num:
                max_points_num = len(box)
        ex_boxes = []
        for box in boxes:
            ex_box = box + [box[-1]] * (max_points_num - len(box))
            ex_boxes.append(ex_box)
        return ex_boxes

    def __call__(self, data):
        """
        required keys:
            label (str): string containgin points and transcription in json format
        added keys:
            polys (np.ndarray): polygon boxes in an image, each polygon is represented by points
                            in shape [num_polygons, num_points, 2]
            texts (List(str)): text string
            ignore_tags (np.ndarray[bool]): indicators for ignorable texts (e.g., '###')
        """
        label = data['label']
        label = json.loads(label)
        nBox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(0, nBox):
            box = label[bno]['points']
            txt = label[bno]['transcription']
            boxes.append(box)
            txts.append(txt)
            if txt in ['*', '###']:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        if len(boxes) == 0:
            return None
        boxes = self.expand_points_num(boxes)
        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=np.bool)

        data['polys'] = boxes
        data['texts'] = txts
        data['ignore_tags'] = txt_tags
        return data


# FIXME:
#  RuntimeWarning: invalid value encountered in sqrt result = np.sqrt(a_sq * b_sq * sin_sq / c_sq)
#  RuntimeWarning: invalid value encountered in true_divide cos = (a_sq + b_sq - c_sq) / (2 * np.sqrt(a_sq * b_sq))
warnings.filterwarnings("ignore")
class BorderMap:
    def __init__(self, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7):
        self._thresh_min = thresh_min
        self._thresh_max = thresh_max
        self._dist_coef = 1 - shrink_ratio ** 2

    def __call__(self, data):
        border = np.zeros(data['image'].shape[:2], dtype=np.float32)
        mask = np.zeros(data['image'].shape[:2], dtype=np.float32)

        for i in range(len(data['polys'])):
            if not data['ignore_tags'][i]:
                self._draw_border(data['polys'][i], border, mask=mask)
        border = border * (self._thresh_max - self._thresh_min) + self._thresh_min

        data['thresh_map'] = border
        data['thresh_mask'] = mask
        return data

    def _draw_border(self, np_poly, border, mask):
        # draw mask
        poly = Polygon(np_poly)
        distance = self._dist_coef * poly.area / poly.length
        padded_polygon = np.array(expand_poly(np_poly, distance)[0], dtype=np.int32)
        cv2.fillPoly(mask, [padded_polygon], 1.0)

        # draw border
        min_vals, max_vals = np.min(padded_polygon, axis=0), np.max(padded_polygon, axis=0)
        width, height = max_vals - min_vals + 1
        np_poly -= min_vals

        xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = [self._distance(xs, ys, p1, p2) for p1, p2 in zip(np_poly, np.roll(np_poly, 1, axis=0))]
        distance_map = np.clip(np.array(distance_map, dtype=np.float32) / distance, 0, 1).min(axis=0)   # NOQA

        min_valid = np.clip(min_vals, 0, np.array(border.shape[::-1]) - 1)  # shape reverse order: w, h
        max_valid = np.clip(max_vals, 0, np.array(border.shape[::-1]) - 1)

        border[min_valid[1]: max_valid[1] + 1, min_valid[0]: max_valid[0] + 1] = np.fmax(
            1 - distance_map[min_valid[1] - min_vals[1]: max_valid[1] - max_vals[1] + height,
                             min_valid[0] - min_vals[0]: max_valid[0] - max_vals[0] + width],
            border[min_valid[1]: max_valid[1] + 1, min_valid[0]: max_valid[0] + 1]
        )

    @staticmethod
    def _distance(xs, ys, point_1, point_2):
        """
        compute the distance from each point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        """
        a_sq = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        b_sq = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        c_sq = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cos = (a_sq + b_sq - c_sq) / (2 * np.sqrt(a_sq * b_sq))
        sin_sq = np.nan_to_num(1 - np.square(cos))
        result = np.sqrt(a_sq * b_sq * sin_sq / c_sq)

        result[cos >= 0] = np.sqrt(np.fmin(a_sq, b_sq))[cos >= 0]
        return result


class ShrinkBinaryMap:
    """
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    """
    def __init__(self, min_text_size=8, shrink_ratio=0.4):
        self._min_text_size = min_text_size
        self._dist_coef = 1 - shrink_ratio ** 2

    def __call__(self, data):
        gt = np.zeros(data['image'].shape[:2], dtype=np.float32)
        mask = np.ones(data['image'].shape[:2], dtype=np.float32)

        if len(data['polys']):
            for i in range(len(data['polys'])):
                min_side = min(np.max(data['polys'][i], axis=0) - np.min(data['polys'][i], axis=0))

                if data['ignore_tags'][i] or min_side < self._min_text_size:
                    cv2.fillPoly(mask, [data['polys'][i].astype(np.int32)], 0)
                    data['ignore_tags'][i] = True
                else:
                    poly = Polygon(data['polys'][i])
                    shrunk = expand_poly(data['polys'][i], distance=-self._dist_coef * poly.area / poly.length)

                    if shrunk:
                        cv2.fillPoly(gt, [np.array(shrunk[0], dtype=np.int32)], 1)
                    else:
                        cv2.fillPoly(mask, [data['polys'][i].astype(np.int32)], 0)
                        data['ignore_tags'][i] = True

        data['binary_map'] = np.expand_dims(gt, axis=0)
        data['mask'] = mask
        return data


class DetResize(object):
    """
    Resize the image and text polygons (if have) for text detection

    Args:
        target_size: target size [H, W] of the output image. If it is not None, `limit_type` will be forced to None and side limit-based resizng will not make effect. Default: None.
        keep_ratio: whether to keep aspect ratio. Default: True
        padding: whether to pad the image to the `target_size` after "keep-ratio" resizing. Only used when keep_ratio is True. Default False.
        limit_type: it decides the resize method type. Option: 'min', 'max', None. Default: "min"
            - 'min': images will be resized by limiting the mininum side length to `limit_side_len`, i.e., any side of the image must be larger than or equal to `limit_side_len`. If the input image alreay fulfill this limitation, no scaling will performed. If not, input image will be up-scaled with the ratio of (limit_side_len / shorter side length)
            - 'max': images will be resized by limiting the maximum side length to `limit_side_len`, i.e., any side of the image must be smaller than or equal to `limit_side_len`. If the input image alreay fulfill this limitation, no scaling will performed. If not, input image will be down-scaled with the ratio of (limit_side_len / longer side length)
            -  None: No limitation. Images will be resized to `target_size` with or without `keep_ratio` and `padding`
        limit_side_len: side len limitation.
        force_divisable: whether to force the image being resize to a size multiple of `divisor` (e.g. 32) in the end, which is suitable for some networks (e.g. dbnet-resnet50). Default: True.
        divisor: divisor used when `force_divisable` enabled. The value is decided by the down-scaling path of the network backbone (e.g. resnet, feature map size is 2^5 smaller than input image size). Default is 32.
        interpoloation: interpolation method

    Note:
        1. The default choices limit_type=min, with large `limit_side_len` are recommended for inference in detection for better accuracy,
        2. If target_size set, keep_ratio=True, limit_type=null, padding=True, this transform works the same as ScalePadImage,
        3. If inference speed is the first priority to guarante, you can set limit_type=max with a small `limit_side_len` like 960.
    """
    def __init__(self,
                 target_size: list = None,
                 keep_ratio=True,
                 padding=False,
                 limit_type='min',
                 limit_side_len=736,
                 force_divisable = True,
                 divisor = 32,
                 interpolation=cv2.INTER_LINEAR):

        if target_size is not None:
            limit_type = None

        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.padding = padding
        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
        self.interpolation = interpolation
        self.force_divisable = force_divisable
        self.divisor = divisor

        if limit_type in ['min', 'max']:
            keep_ratio = True
            padding = False
            print('INFO: `limit_type` is {limit_type}. Image will be resized by limiting the {limit_type} side length to {limit_side_len}.')
        elif not limit_type:
            assert target_size is not None or force_divisable is not None, 'One of `target_size` or `force_divisable` is required when limit_type is not set. Please set at least one of them.'
            if target_size and force_divisable:
                if (target_size[0] % divisor != 0) or (target_size[1] % divisor != 0):
                    self.target_size= [max(round(x / self.divisor) * self.divisor, self.divisor) for x in target_size]
                    print(f'WARNING: `force_divisable` is enabled but the set target size {target_size} is not divisable by {divisor}. Target size is ajusted to {self.target_size}')
            if (target_size is not None) and keep_ratio and (not padding):
                print(f'WARNING: output shape can be dynamic if keep_ratio but no padding.')
        else:
            raise ValueError(f'Unknown limit_type: {limit_type}')

    def __call__(self, data: dict):
        """
        required keys:
            image: shape HWC
            polys: shape [num_polys, num_points, 2] (optional)
        modified keys:
            image
            (polys)
        added keys:
            shape: [src_h, src_w, scale_ratio_h, scale_ratio_w]
        """
        img = data['image']
        h, w = img.shape[:2]
        if self.target_size:
            tar_h, tar_w = self.target_size

        scale_ratio = 1.0
        allow_padding = False
        if self.limit_type == 'min':
            if min(h, w) < self.limit_side_len: # upscale
                scale_ratio = self.limit_side_len / float(min(h, w))
        elif self.limit_type == 'max':
            if max(h, w) > self.limit_side_len: # downscale
                scale_ratio = self.limit_side_len / float(max(h, w))
        elif not self.limit_type:
            if self.keep_ratio:
                # scale the image until it fits in the target size at most. The left part could be filled by padding.
                scale_ratio = min(tar_h / h, tar_w / w)
                allow_padding = True

        if (self.limit_type in ['min', 'max']) or self.keep_ratio:
            resize_w = math.ceil(w * scale_ratio)
            resize_h = math.ceil(h * scale_ratio)
        else:
            resize_w = tar_w
            resize_h = tar_h

        if self.force_divisable:
            if not (allow_padding and self.padding): # no need to round it the image will be padded to the target size which is divisable.
                # adjust the size slightly so that both sides of the image are divisable by divisor e.g. 32, which could be required by the network
                resize_h = max(round(resize_h / self.divisor) * self.divisor, self.divisor)
                resize_w = max(round(resize_w / self.divisor) * self.divisor, self.divisor)

        resized_img = cv2.resize(img, (resize_w, resize_h), interpolation=self.interpolation)

        if allow_padding and self.padding:
            if self.target_size and (tar_h >= resize_h and tar_w >= resize_w):
                padded_img = np.zeros((tar_h, tar_w, 3), dtype=np.uint8)
                padded_img[:resize_h, :resize_w, :] = resized_img
                data['image'] = padded_img
            else:
                raise ValueError(f'`target_size` must be set to be not smaller than (resize_h, resize_w) for padding, but found {self.target_size}')
        else:
            data['image'] = resized_img

        scale_h = resize_h / h
        scale_w = resize_w / w
        if 'polys' in data:
            data['polys'][:, :, 0] = data['polys'][:, :, 0] * scale_w
            data['polys'][:, :, 1] = data['polys'][:, :, 1] * scale_h
        data['shape'] = [h, w, scale_h, scale_w]

        return data


def expand_poly(poly, distance: float, joint_type=pyclipper.JT_ROUND) -> List[list]:
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(poly, joint_type, pyclipper.ET_CLOSEDPOLYGON)
    return offset.Execute(distance)
