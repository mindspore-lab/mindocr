"""
transforms for text detection tasks.
"""
import json
import math
import random
import warnings
from typing import List, Tuple

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon, box

from mindocr.utils.logger import Logger

__all__ = [
    "DetLabelEncode",
    "BorderMap",
    "ShrinkBinaryMap",
    "expand_poly",
    "PSEGtDecode",
    "ValidatePolygons",
    "RandomCropWithBBox",
    "RandomCropWithMask",
    "DetResize",
    "GridResize",
    "ScalePadImage",
]
_logger = Logger("mindocr")


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
        for b in boxes:
            if len(b) > max_points_num:
                max_points_num = len(b)
        ex_boxes = []
        for b in boxes:
            ex_box = b + [b[-1]] * (max_points_num - len(b))
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
        label = data["label"]
        label = json.loads(label)
        nBox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(0, nBox):
            box = label[bno]["points"]
            txt = label[bno]["transcription"]
            boxes.append(box)
            txts.append(txt)
            if txt in ["*", "###"]:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        if len(boxes) == 0:
            return None
        boxes = self.expand_points_num(boxes)
        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=np.bool)

        data["polys"] = boxes
        data["texts"] = txts
        data["ignore_tags"] = txt_tags
        return data


class RandomCropWithBBox:
    """
    Randomly cuts a crop from an image along with polygons in the way that the crop doesn't intersect any polygons
    (i.e. any given polygon is either fully inside or fully outside the crop).

    Args:
        max_tries: number of attempts to try to cut a crop with a polygon in it. If fails, scales the whole image to
                   match the `crop_size`.
        min_crop_ratio: minimum size of a crop in respect to an input image size.
        crop_size: target size of the crop (resized and padded, if needed), preserves sides ratio.
        p: probability of the augmentation being applied to an image.
    """

    def __init__(
        self, max_tries: int = 10, min_crop_ratio: float = 0.1, crop_size: Tuple = (640, 640), p: float = 0.5, **kwargs
    ):
        self._crop_size = crop_size
        self._ratio = min_crop_ratio
        self._max_tries = max_tries
        self._p = p

    def __call__(self, data: dict) -> dict:
        if random.random() < self._p:  # cut a crop
            start, end = self._find_crop(data)
        else:  # scale and pad the whole image
            start, end = np.array([0, 0]), np.array(data["image"].shape[:2])

        scale = min(self._crop_size / (end - start))

        data["image"] = cv2.resize(data["image"][start[0] : end[0], start[1] : end[1]], None, fx=scale, fy=scale)
        data["actual_size"] = np.array(data["image"].shape[:2])
        data["image"] = np.pad(
            data["image"], (*tuple((0, cs - ds) for cs, ds in zip(self._crop_size, data["image"].shape[:2])), (0, 0))
        )

        data["polys"] = (data["polys"] - start[::-1]) * scale

        return data

    def _find_crop(self, data: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find a suitable crop that doesn't intersect any polygon by randomly sampling possible coordinates.

        Args:
            data: dict containing image, polys and ignore_tags keys.

        Returns:
            Tuple: start and end coordinates of the crop.
        """
        size = np.array(data["image"].shape[:2])
        polys = [poly for poly, ignore in zip(data["polys"], data["ignore_tags"]) if not ignore]

        if polys:
            # do not crop through polys => find available "empty" coordinates
            h_array, w_array = np.zeros(size[0], dtype=np.int32), np.zeros(size[1], dtype=np.int32)
            for poly in polys:
                points = np.maximum(np.round(poly).astype(np.int32), 0)
                w_array[points[:, 0].min() : points[:, 0].max() + 1] = 1
                h_array[points[:, 1].min() : points[:, 1].max() + 1] = 1

            if not h_array.all() and not w_array.all():  # if texts do not occupy full image
                # find available coordinates that don't include text
                h_avail = np.where(h_array == 0)[0]
                w_avail = np.where(w_array == 0)[0]

                min_size = np.ceil(size * self._ratio).astype(np.int32)
                for _ in range(self._max_tries):
                    y = np.sort(np.random.choice(h_avail, size=2))
                    x = np.sort(np.random.choice(w_avail, size=2))
                    start, end = np.array([y[0], x[0]]), np.array([y[1], x[1]])

                    if ((end - start) < min_size).any():  # NOQA
                        continue

                    # check that at least one polygon is within the crop
                    for poly in polys:
                        if (poly.max(axis=0) > start[::-1]).all() and (poly.min(axis=0) < end[::-1]).all():  # NOQA
                            return start, end

        # failed to generate a crop or all polys are marked as ignored
        return np.array([0, 0]), size


class RandomCropWithMask:
    def __init__(self, size, main_key, crop_keys, p=3 / 8, **kwargs):
        self.size = size
        self.main_key = main_key
        self.crop_keys = crop_keys
        self.p = p

    def __call__(self, data):
        image = data["image"]

        h, w = image.shape[0:2]
        th, tw = self.size
        if w == tw and h == th:
            return data

        mask = data[self.main_key]
        if np.max(mask) > 0 and np.random.random() > self.p:
            # make sure to crop the text region
            tl = np.min(np.where(mask > 0), axis=1) - (th, tw)
            tl[tl < 0] = 0
            br = np.max(np.where(mask > 0), axis=1) - (th, tw)
            br[br < 0] = 0

            br[0] = min(br[0], h - th)
            br[1] = min(br[1], w - tw)

            i = np.random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
            j = np.random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
        else:
            i = np.random.randint(0, h - th) if h - th > 0 else 0
            j = np.random.randint(0, w - tw) if w - tw > 0 else 0

        # return i, j, th, tw
        for k in data:
            if k in self.crop_keys:
                if len(data[k].shape) == 3:
                    if np.argmin(data[k].shape) == 0:
                        img = data[k][:, i : i + th, j : j + tw]
                    elif np.argmin(data[k].shape) == 2:
                        img = data[k][i : i + th, j : j + tw, :]
                    else:
                        img = data[k]
                else:
                    img = data[k][i : i + th, j : j + tw]
                data[k] = img
        return data


# FIXME: BorderMap
#  RuntimeWarning: invalid value encountered in sqrt result = np.sqrt(a_sq * b_sq * sin_sq / c_sq)
#  RuntimeWarning: invalid value encountered in true_divide cos = (a_sq + b_sq - c_sq) / (2 * np.sqrt(a_sq * b_sq))
warnings.filterwarnings("ignore")


class BorderMap:
    """
    Generate a border (or threshold) map along text polygon edges.

    Args:
        shrink_ratio: polygon shrink ratio (same as in ShrinkBinaryMap) which is used to calculate width of the border.
        thresh_min: minimum value for the border map (normalized map).
        thresh_max: maximum value for the border map (normalized map).
    """

    def __init__(self, shrink_ratio: float = 0.4, thresh_min: float = 0.3, thresh_max: float = 0.7, **kwargs):
        self._thresh_min = thresh_min
        self._thresh_max = thresh_max
        self._dist_coef = 1 - shrink_ratio**2

    def __call__(self, data: dict) -> dict:
        border = np.zeros(data["image"].shape[:2], dtype=np.float32)
        mask = np.zeros(data["image"].shape[:2], dtype=np.float32)

        for i in range(len(data["polys"])):
            if not data["ignore_tags"][i]:
                self._draw_border(data["polys"][i], border, mask=mask)
        border = border * (self._thresh_max - self._thresh_min) + self._thresh_min

        data["thresh_map"] = border
        data["thresh_mask"] = mask
        return data

    def _draw_border(self, np_poly: np.ndarray, border: np.ndarray, mask: np.ndarray):
        # draw mask
        poly = Polygon(np_poly)
        distance = self._dist_coef * poly.area / poly.length
        padded_polygon = np.array(expand_poly(np_poly, distance)[0], dtype=np.int32)
        cv2.fillPoly(mask, [padded_polygon], 1.0)

        # draw border
        min_vals, max_vals = np.min(padded_polygon, axis=0), np.max(padded_polygon, axis=0)
        width, height = max_vals - min_vals + 1
        np_poly = np_poly - min_vals

        xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = [self._distance(xs, ys, p1, p2) for p1, p2 in zip(np_poly, np.roll(np_poly, 1, axis=0))]
        distance_map = np.clip(np.array(distance_map, dtype=np.float32) / distance, 0, 1).min(axis=0)  # NOQA

        min_valid = np.clip(min_vals, 0, np.array(border.shape[::-1]) - 1)  # shape reverse order: w, h
        max_valid = np.clip(max_vals, 0, np.array(border.shape[::-1]) - 1)

        border[min_valid[1] : max_valid[1] + 1, min_valid[0] : max_valid[0] + 1] = np.fmax(
            1
            - distance_map[
                min_valid[1] - min_vals[1] : max_valid[1] - max_vals[1] + height,
                min_valid[0] - min_vals[0] : max_valid[0] - max_vals[0] + width,
            ],
            border[min_valid[1] : max_valid[1] + 1, min_valid[0] : max_valid[0] + 1],
        )

    @staticmethod
    def _distance(xs: np.ndarray, ys: np.ndarray, point_1: np.ndarray, point_2: np.ndarray) -> np.ndarray:
        """
        Compute distance from each point in an image to a specified edge.

        Args:
            xs: x-axis values
            ys: y-axis values
            point_1 (x, y): start coordinate of the edge.
            point_2 (x, y): end coordinate of the edge.

        Returns:
            border map for a single edge.
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
    Generate a shrunk binary mask from detection labels. Typically, follows the process of class `MakeICDARData`.

    Args:
        min_text_size: minimum text size (in pixel) below which the label is marked as 'ignore'.
        shrink_ratio: text mask shrinkage ratio.
    """

    def __init__(self, min_text_size: int = 8, shrink_ratio: int = 0.4, **kwargs):
        self._min_text_size = min_text_size
        self._dist_coef = 1 - shrink_ratio**2

    def __call__(self, data: dict) -> dict:
        gt = np.zeros(data["image"].shape[:2], dtype=np.float32)
        mask = np.ones(data["image"].shape[:2], dtype=np.float32)

        if len(data["polys"]):
            for i in range(len(data["polys"])):
                min_side = min(np.max(data["polys"][i], axis=0) - np.min(data["polys"][i], axis=0))

                if data["ignore_tags"][i] or min_side < self._min_text_size:
                    cv2.fillPoly(mask, [data["polys"][i].astype(np.int32)], 0)
                    data["ignore_tags"][i] = True
                else:
                    poly = Polygon(data["polys"][i])
                    shrunk = expand_poly(data["polys"][i], distance=-self._dist_coef * poly.area / poly.length)

                    if shrunk:
                        cv2.fillPoly(gt, [np.array(shrunk[0], dtype=np.int32)], 1)
                    else:
                        cv2.fillPoly(mask, [data["polys"][i].astype(np.int32)], 0)
                        data["ignore_tags"][i] = True

        data["binary_map"] = np.expand_dims(gt, axis=0)
        data["mask"] = mask
        return data


class DetResize:
    """
    Resize the image and text polygons (if any) for text detection

    Args:
        target_size: target size [H, W] of the output image. If it is not None, `limit_type` will be forced to None and
            side limit-based resizing will not make effect. Default: None.
        keep_ratio: whether to keep aspect ratio. Default: True
        padding: whether to pad the image to the `target_size` after "keep-ratio" resizing. Only used when keep_ratio is
            True. Default False.
        limit_type: it decides the resize method type. Option: 'min', 'max', None. Default: "min"
            - 'min': images will be resized by limiting the minimum side length to `limit_side_len`, i.e.,
              any side of the image must be larger than or equal to `limit_side_len`. If the input image alreay fulfill
            this limitation, no scaling will be performed. If not, input image will be up-scaled with the ratio of
            (limit_side_len / shorter side length)
            - 'max': images will be resized by limiting the maximum side length to `limit_side_len`, i.e.,
              any side of the image must be smaller than or equal to `limit_side_len`. If the input image alreay fulfill
              this limitation, no scaling will be performed. If not, input image will be down-scaled with the ratio of
              (limit_side_len / longer side length)
            -  None: No limitation. Images will be resized to `target_size` with or without `keep_ratio` and `padding`
        limit_side_len: side len limitation.
        force_divisable: whether to force the image being resized to a size multiple of `divisor` (e.g. 32) in the end,
            which is suitable for some networks (e.g. dbnet-resnet50). Default: True.
        divisor: divisor used when `force_divisable` enabled. The value is decided by the down-scaling path of
            the network backbone (e.g. resnet, feature map size is 2^5 smaller than input image size). Default is 32.
        interpolation: interpolation method

    Note:
        1. The default choices limit_type=min, with large `limit_side_len` are recommended for inference in detection
        for better accuracy,
        2. If target_size set, keep_ratio=True, limit_type=null, padding=True, this transform works the same as
        ScalePadImage,
        3. If inference speed is the first priority to guarantee, you can set limit_type=max with a small
        `limit_side_len` like 960.
    """

    def __init__(
        self,
        target_size: list = None,
        keep_ratio: bool = True,
        padding: bool = False,
        limit_type: str = "min",
        limit_side_len: int = 736,
        force_divisable: bool = True,
        divisor: int = 32,
        interpolation: int = cv2.INTER_LINEAR,
        **kwargs,
    ):
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

        self.is_train = kwargs.get("is_train", False)
        assert target_size is None or limit_type is None, "Only one of limit_type and target_size should be provided."
        if limit_type in ["min", "max"]:
            keep_ratio = True
            padding = False
            _logger.info(
                f"`limit_type` is {limit_type}. Image will be resized by limiting the {limit_type} "
                f"side length to {limit_side_len}."
            )
        elif not limit_type:
            assert target_size is not None or force_divisable is not None, (
                "One of `target_size` or `force_divisable` is required when limit_type is not set. "
                "Please set at least one of them."
            )
            if target_size and force_divisable:
                if (target_size[0] % divisor != 0) or (target_size[1] % divisor != 0):
                    self.target_size = [max(round(x / self.divisor) * self.divisor, self.divisor) for x in target_size]
                    _logger.warning(
                        f"`force_divisable` is enabled but the set target size {target_size} "
                        f"is not divisable by {divisor}. Target size is ajusted to {self.target_size}"
                    )
            if (target_size is not None) and keep_ratio and (not padding):
                _logger.warning("output shape can be dynamic if keep_ratio but no padding.")
        else:
            raise ValueError(f"Unknown limit_type: {limit_type}")

    def __call__(self, data: dict) -> dict:
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
        img = data["image"]
        h, w = img.shape[:2]
        if self.target_size:
            tar_h, tar_w = self.target_size

        scale_ratio = 1.0
        allow_padding = False
        if self.limit_type == "min":
            if min(h, w) < self.limit_side_len:  # upscale
                scale_ratio = self.limit_side_len / float(min(h, w))
        elif self.limit_type == "max":
            if max(h, w) > self.limit_side_len:  # downscale
                scale_ratio = self.limit_side_len / float(max(h, w))
        elif not self.limit_type:
            if self.keep_ratio and self.target_size:
                # scale the image until it fits in the target size at most. The left part could be filled by padding.
                scale_ratio = min(tar_h / h, tar_w / w)
                allow_padding = True

        if (self.limit_type in ["min", "max"]) or (self.target_size and self.keep_ratio):
            resize_w = math.ceil(w * scale_ratio)
            resize_h = math.ceil(h * scale_ratio)
            if self.target_size:
                resize_w = min(resize_w, tar_w)
                resize_h = min(resize_h, tar_h)
        elif self.target_size:
            resize_w = tar_w
            resize_h = tar_h
        else:  # both target_size and limit_type is None. resize by force_divisable
            resize_w = w
            resize_h = h

        if self.force_divisable:
            if not (
                allow_padding and self.padding
            ):  # no need to round it the image will be padded to the target size which is divisable.
                # adjust the size slightly so that both sides of the image are divisable by divisor
                # e.g. 32, which could be required by the network
                resize_h = max(
                    math.ceil(resize_h / self.divisor) * self.divisor, self.divisor
                )  # diff from resize_image_type0 in pp which uses round()
                resize_w = max(math.ceil(resize_w / self.divisor) * self.divisor, self.divisor)

        resized_img = cv2.resize(img, (resize_w, resize_h), interpolation=self.interpolation)

        if allow_padding and self.padding:
            if self.target_size and (tar_h >= resize_h and tar_w >= resize_w):
                # do padding
                padded_img = np.zeros((tar_h, tar_w, 3), dtype=np.uint8)
                padded_img[:resize_h, :resize_w, :] = resized_img
                data["image"] = padded_img
            else:
                _logger.warning(
                    f"Image shape after resize is ({resize_h}, {resize_w}), "
                    f"which is larger than target_size {self.target_size}. Skip padding for the current image. "
                    f"You may disable `force_divisable` to avoid this warning."
                )
        else:
            data["image"] = resized_img

        scale_h = resize_h / h
        scale_w = resize_w / w

        # Only need to transform ground truth polygons in training for generating masks/maps.
        # For evaluation, we should not change the GT polygons.
        # The metric with input of GT polygons and predicted polygons must be computed in the original image space
        # for consistent comparison.
        if "polys" in data and self.is_train:
            data["polys"][:, :, 0] = data["polys"][:, :, 0] * scale_w
            data["polys"][:, :, 1] = data["polys"][:, :, 1] * scale_h

        if "shape_list" not in data:
            src_h, src_w = data.get("raw_img_shape", (h, w))
            data["shape_list"] = np.array([src_h, src_w, scale_h, scale_w], dtype=np.float32)
        else:
            data["shape_list"][2] = data["shape_list"][2] * scale_h
            data["shape_list"][3] = data["shape_list"][3] * scale_h

        return data


class GridResize(DetResize):
    """
    Resize image to make it divisible by a specified factor exactly.
    Resize polygons correspondingly, if provided.

    Args:
        factor: by which an image should be divisible.
    """

    def __init__(self, factor: int = 32, **kwargs):
        super().__init__(
            target_size=None,
            keep_ratio=False,
            padding=False,
            limit_type=None,
            force_divisable=True,
            divisor=factor,
        )


class ScalePadImage(DetResize):
    """
    Scale image and polys by the shorter side, then pad to the target_size.
    input image format: hwc

    Args:
        target_size: [H, W] of the output image.
    """

    def __init__(self, target_size: list, **kwargs):
        super().__init__(
            target_size=target_size,
            keep_ratio=True,
            padding=True,
            limit_type=None,
            force_divisable=False,
        )


def expand_poly(poly, distance: float, joint_type=pyclipper.JT_ROUND) -> List[list]:
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(poly, joint_type, pyclipper.ET_CLOSEDPOLYGON)
    return offset.Execute(distance)


class PSEGtDecode:
    """
    PSENet transformation which shrinks text polygons.

    Args:
       kernel_num (int): The number of kernels.
       min_shrink_ratio (float): The minimum shrink ratio.
       min_shortest_edge (int): The minimum shortest edge.

    Returns:
       dict: A dictionary containing shrinked image data, polygons, ground truth kernels, ground truth text and masks.
    """

    def __init__(self, kernel_num=7, min_shrink_ratio=0.4, min_shortest_edge=640, **kwargs):
        self.kernel_num = kernel_num
        self.min_shrink_ratio = min_shrink_ratio
        self.min_shortest_edge = min_shortest_edge

    def _shrink(self, text_polys, rate, max_shr=20):
        """
        Shrink text polygons.

        Args:
            text_polys (list): A list of text polygons.
            rate (float): The shrink rate.
            max_shr (int): The maximum shrink.

        Returns:
            list: A list of shrinked text polygons.
        """
        rate = rate * rate
        shrinked_text_polys = []
        for bbox in text_polys:
            poly = Polygon(bbox)
            area, peri = poly.area, poly.length

            offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)
            shrinked_bbox = expand_poly(bbox, -offset)  # (N, 2) shape, N maybe larger than or smaller than 4.
            if not shrinked_bbox:
                shrinked_text_polys.append(bbox)
                continue

            shrinked_bbox = np.array(shrinked_bbox)[0]
            shrinked_bbox = np.array(shrinked_bbox)
            if shrinked_bbox.shape[0] <= 2:
                shrinked_text_polys.append(bbox)
                continue

            shrinked_text_polys.append(shrinked_bbox)

        return shrinked_text_polys

    def __call__(self, data):
        """
        Args:
            data (dict): A dictionary containing image data.

        Returns:
            dict: dict: A dictionary containing shrinked image data, polygons,
            ground truth kernels, ground truth text and masks.
        """
        image = data["image"]
        text_polys = data["polys"]
        ignore_tags = data["ignore_tags"]

        h, w, _ = image.shape
        short_edge = min(h, w)
        if short_edge < self.min_shortest_edge:
            # keep short_size >= self.min_short_edge
            scale = self.min_shortest_edge / short_edge
            image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
            text_polys *= scale

        # get gt_text and training_mask
        img_h, img_w = image.shape[0:2]
        gt_text = np.zeros((img_h, img_w), dtype=np.float32)
        training_mask = np.ones((img_h, img_w), dtype=np.float32)
        if text_polys.shape[0] > 0:
            text_polys = text_polys.astype("int32")
            for i in range(text_polys.shape[0]):
                cv2.drawContours(gt_text, [text_polys[i]], 0, i + 1, -1)
                if ignore_tags[i]:
                    cv2.drawContours(training_mask, [text_polys[i]], 0, 0, -1)

        # get gt_kernels
        gt_kernels = []
        for i in range(1, self.kernel_num):
            rate = 1.0 - (1.0 - self.min_shrink_ratio) / (self.kernel_num - 1) * i
            gt_kernel = np.zeros((img_h, img_w), dtype=np.float32)
            kernel_text_polys = self._shrink(text_polys, rate)
            for j in range(len(kernel_text_polys)):
                cv2.drawContours(gt_kernel, [kernel_text_polys[j]], 0, 1, -1)
            gt_kernels.append(gt_kernel)

        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        data["image"] = image
        data["polys"] = text_polys
        data["gt_kernels"] = gt_kernels
        data["gt_text"] = gt_text
        data["mask"] = training_mask
        return data


class ValidatePolygons:
    """
    Validate polygons by:
     1. filtering out polygons outside an image.
     2. clipping coordinates of polygons that are partially outside an image to stay within the visible region.
    Args:
        min_area: minimum area below which newly clipped polygons considered as ignored.
        clip_to_visible_area: (Experimental) clip polygons to a visible area. Number of vertices in a polygon after
                              clipping may change.
        min_vertices: minimum number of vertices in a polygon below which newly clipped polygons considered as ignored.
    """

    def __init__(self, min_area: float = 1.0, clip_to_visible_area: bool = False, min_vertices: int = 4, **kwargs):
        self._min_area = min_area
        self._min_vert = min_vertices
        self._clip = clip_to_visible_area

    def __call__(self, data: dict) -> dict:
        size = data.get("actual_size", np.array(data["image"].shape[:2]))[::-1]  # convert to x, y coord
        border = box(0, 0, *size)

        new_polys, new_texts, new_tags = [], [], []
        for np_poly, text, ignore in zip(data["polys"], data["texts"], data["ignore_tags"]):
            poly = Polygon(np_poly)
            if poly.intersects(border):  # if the polygon is fully or partially within the image
                poly = poly.intersection(border)
                if poly.area < self._min_area:
                    ignore = True

                if self._clip:  # Clip polygon to a visible area
                    poly = poly.exterior.coords
                    np_poly = np.array(poly[:-1])
                    if len(np_poly) < self._min_vert:
                        ignore = True

                new_polys.append(np_poly)
                new_tags.append(ignore)
                new_texts.append(text)

        data["polys"] = new_polys
        data["texts"] = new_texts
        data["ignore_tags"] = np.array(new_tags)

        return data
