import random
from typing import Union

import cv2
import numpy as np

__all__ = [
    "RandomScale",
    "RandomRotate",
    "RandomHorizontalFlip",
]


class RandomScale:
    """
    Randomly scales an image and its polygons in a predefined scale range.
    Args:
        scale_range: (min, max) scale range.
        size_limits: (min_side_len, max_side_len) size limits. Default: None.
        p: probability of the augmentation being applied to an image.
    """

    def __init__(
        self,
        scale_range: Union[tuple, list],
        size_limits: Union[tuple, list] = None,
        polygons=True,
        p: float = 0.5,
        **kwargs,
    ):
        self._range = sorted(scale_range)
        self._size_limits = sorted(size_limits) if size_limits else []
        self._p = p
        self.output_columns = ["image", "polys"] if polygons else ["image"]
        assert kwargs.get("is_train", True), ValueError("RandomScale augmentation must be used for training only")

    def __call__(self, data: dict) -> dict:
        """
        required keys:
            image, HWC
            (polys)
        modified keys:
            image
            (polys)
        """
        if random.random() < self._p:
            if self._size_limits:
                size = data["image"].shape[:2]
                min_scale = max(self._size_limits[0] / size[0], self._size_limits[0] / size[1], self._range[0])
                max_scale = min(self._size_limits[1] / size[0], self._size_limits[1] / size[1], self._range[1])
                scale = np.random.uniform(min_scale, max_scale)
            else:
                scale = np.random.uniform(*self._range)

            data["image"] = cv2.resize(data["image"], dsize=None, fx=scale, fy=scale)

            if "polys" in self.output_columns:
                data["polys"] *= scale
        return data


class RandomRotate:
    """
    Randomly rotate an image with polygons in it (if any).
    Args:
        degrees: range of angles [min, max]
        expand_canvas: whether to expand canvas during rotation (the image size will be increased) or
                       maintain the original size (the rotated image will be cropped back to the original size).
        polygons: apply rotation to polygons as well
        p: probability of the augmentation being applied to an image.
    """

    def __init__(self, degrees=(-10, 10), expand_canvas=True, polygons: bool = True, p: float = 1.0, **kwargs):
        self._degrees = degrees
        self._canvas = expand_canvas
        self._p = p
        self.output_columns = ["image", "polys"] if polygons else ["image"]

    def __call__(self, data: dict) -> dict:
        if random.random() < self._p:
            angle = random.randint(self._degrees[0], self._degrees[1])
            h, w = data["image"].shape[:2]

            center = w // 2, h // 2  # x, y
            mat = cv2.getRotationMatrix2D(center, angle, 1)

            if self._canvas:
                # compute the new bounding dimensions of the image
                cos, sin = np.abs(mat[0, 0]), np.abs(mat[0, 1])
                w, h = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))

                # adjust the rotation matrix to take into account translation
                mat[0, 2] += (w / 2) - center[0]
                mat[1, 2] += (h / 2) - center[1]

            data["image"] = cv2.warpAffine(data["image"], mat, (w, h))

            if "polys" in self.output_columns and len(data["polys"]):
                data["polys"] = cv2.transform(data["polys"], mat)

        return data


class RandomHorizontalFlip:
    """
    Random horizontal flip of an image with polygons in it (if any).
    Args:
        p: probability of the augmentation being applied to an image.
    """

    def __init__(self, polygons: bool = True, p: float = 0.5, **kwargs):
        self._p = p
        self.output_columns = ["image", "polys"] if polygons else ["image"]

    def __call__(self, data: dict) -> dict:
        if random.random() < self._p:
            data["image"] = cv2.flip(data["image"], 1)

            if "polys" in self.output_columns and len(data["polys"]):
                mat = np.float32([[-1, 0, data["image"].shape[1] - 1], [0, 1, 0]])
                data["polys"] = cv2.transform(data["polys"], mat)
                # TODO: assign a new starting point located in the top left
                data["polys"] = data["polys"][:, ::-1, :]  # preserve the original order (e.g. clockwise)

        return data
