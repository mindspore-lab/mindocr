import cv2
import numpy as np

from mindspore import Tensor

from .det_base_postprocess import DetBasePostprocess

__all__ = ["PSEPostprocess"]


def _resize_4d_array(pred, scale_factor):
    """
    Resize a 4D numpy array using bilinear interpolation in the H and W dimensions.
    :param pred: A 4D numpy array of shape (N, C, H, W).
    :param scale_factor: An integer value representing the scale factor applied to H and W.
    :return: A 4D numpy array of shape (N, C, H', W'), where H' and W' are scaled versions of H and W.
    """
    N, C, H, W = pred.shape
    scaled_H = H * scale_factor
    scaled_W = W * scale_factor

    # warning: C*N should not exceed 512
    # according to: https://stackoverflow.com/a/65160547/6380135
    pred_3d = np.transpose(pred, axes=(2, 3, 1, 0)).reshape((H, W, C * N))

    resize_3d = cv2.resize(pred_3d, (scaled_W, scaled_H), interpolation=cv2.INTER_LINEAR)

    resized_pred = np.transpose(resize_3d.reshape((scaled_H, scaled_W, C, N)), axes=(3, 2, 0, 1))

    return resized_pred


def _sigmoid_3d_array(x):
    # Compute the sigmoid of the 3D array
    sigmoid_3d = 1 / (1 + np.exp(-x))

    return sigmoid_3d


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

            pred = _resize_4d_array(pred.asnumpy(), scale_factor=4 // self._scale)
            score = _sigmoid_3d_array(pred[:, 0, :, :])
            kernels = (pred > self._binary_thresh).astype(np.float32)
            text_mask = kernels[:, :1, :, :]
            kernels = (kernels * text_mask).astype(np.uint8)

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
