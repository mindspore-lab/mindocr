import cv2
import numpy as np

import mindspore as ms
from mindspore import Tensor, nn

from .det_base_postprocess import DetBasePostprocess

__all__ = ["PSEPostprocess"]


class PSEPostprocess(DetBasePostprocess):
    """
    Post-processing module for PSENet text detection.

    This module takes the network predictions and performs post-processing to obtain the final text detection results.

    Args:
        binary_thresh (float): The threshold value for binarization. Default is 0.5.
        box_thresh (float): The threshold value for generating bounding boxes. Default is 0.85.
        min_area (int): The minimum area threshold for filtering small text regions. Default is 16.
        box_type (str): The type of bounding boxes to generate. Can be "quad" or "poly". Default is "quad".
        scale (int): The scale factor for resizing the predicted output. Default is 4.
        output_score_kernels (bool): Whether to output the scores and kernels. Default is False.
        rescale_fields (list): The list of fields to be rescaled. Default is ["polys"].

    Returns:
        dict: A dictionary containing the final text detection results.
    """

    def __init__(
        self,
        binary_thresh=0.5,
        box_thresh=0.85,
        min_area=16,
        box_type="quad",
        scale=4,
        output_score_kernels=False,
        rescale_fields=["polys"],
    ):
        super().__init__(rescale_fields, box_type)

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
        self._output_score_kernels = output_score_kernels

    def _postprocess(self, pred, **kwargs):  # pred: N 7 H W
        """
        Args:
            pred (Tensor): network prediction with shape [BS, C, H, W]
        """
        score, kernels = None, None
        if self._output_score_kernels:
            score = pred[0]
            kernels = pred[1].astype(np.uint8)
        else:
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
        for batch_idx in range(score.shape[0]):
            boxes, scores = self._boxes_from_bitmap(score[batch_idx], kernels[batch_idx])
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
            elif self._box_type == "poly":
                box_height = np.max(points[:, 1]) + 10
                box_width = np.max(points[:, 0]) + 10
                mask = np.zeros((box_height, box_width), np.uint8)
                mask[points[:, 1], points[:, 0]] = 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bbox = np.squeeze(contours[0], 1)
            else:
                raise NotImplementedError(
                    f"The value of param 'box_type' can only be 'quad', but got '{self._box_type}'."
                )
            boxes.append(bbox)
            scores.append(score_i)

        return boxes, scores
