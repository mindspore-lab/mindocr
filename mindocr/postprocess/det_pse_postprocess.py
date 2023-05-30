from typing import Tuple, Union, List
import cv2
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import nn
from shapely.geometry import Polygon

from .det_base_postprocess import DetBasePostprocess
from ..data.transforms.det_transforms import expand_poly

__all__ = ["PSEPostprocess"]


class PSEPostprocess(DetBasePostprocess):
    def __init__(
        self,
        binary_thresh=0.5,
        box_thresh=0.85,
        min_area=16,
        box_type="quad",
        scale=4,
        rescale_fields=["polys"],
    ):
        super().__init__(box_type, rescale_fields)

        from .pse import pse

        self._binary_thresh = binary_thresh
        self._box_thresh = box_thresh
        self._min_area = min_area
        self._box_type = box_type
        self._scale = scale
        self._interpolate = nn.ResizeBilinear()
        self._sigmoid = nn.Sigmoid()
        if rescale_fields is None:
            rescale_fields = []
        self._rescale_fields = rescale_fields
        self._pse = pse

    def _postprocess(self, pred, **kwargs):  # pred: N 7 H W
        """
        Args:
            pred (Tensor): network prediction with shape [BS, C, H, W]
        """
        if isinstance(pred, tuple):  # used when inference, only need the first output
            pred = pred[0]
        if not isinstance(pred, Tensor):
            pred = Tensor(pred)

        pred = self._interpolate(pred, scale_factor=4 // self._scale)
        score = self._sigmoid(pred[:, 0, :, :])

        kernels = (pred > self._binary_thresh).astype(ms.float32)
        text_mask = kernels[:, :1, :, :]
        text_mask = text_mask.astype(ms.int8)

        kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask
        score = score.asnumpy()
        kernels = kernels.asnumpy().astype(np.uint8)
        poly_list, score_list = [], []
        for batch_idx in range(pred.shape[0]):
            boxes, scores = self._boxes_from_bitmap(
                score[batch_idx], kernels[batch_idx]
            )
            poly_list.append(boxes)
            score_list.append(scores)

        return {"polys": poly_list, "scores": score_list}

    def _boxes_from_bitmap(self, score, kernels):
        label = self._pse(kernels, self._min_area)
        return self._generate_box(score, label)

    def _generate_box(self, score, label):
        label_num = np.max(label) + 1
        boxes = []
        scores = []
        for i in range(1, label_num):
            ind = label == i
            points = np.array(np.where(ind)).transpose((1, 0))[:, ::-1]
            if points.shape[0] < self._min_area:
                label[ind] = 0
                continue

            score_i = np.mean(score[ind])
            if score_i < self._box_thresh:
                label[ind] = 0
                continue

            if self._box_type == "quad":
                rect = cv2.minAreaRect(points)
                bbox = cv2.boxPoints(rect)
            else:
                raise NotImplementedError(
                    f"The value of param 'box_type' can only be 'quad', but got '{self._box_type}'."
                )
            boxes.append(bbox)
            scores.append(score_i)

        return boxes, scores
