from typing import Tuple, Union, List
import cv2
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import nn
from shapely.geometry import Polygon

from .det_base_postprocess import DetBasePostprocess
from ..data.transforms.det_transforms import expand_poly

__all__ = ["DBPostprocess"]


class DBPostprocess(DetBasePostprocess):
    """
    DBNet & DBNet++ postprocessing pipeline: extracts polygons / rectangles from a binary map (heatmap) and returns
        their coordinates.
    Args:
        binary_thresh: binarization threshold applied to the heatmap output of DBNet.
        box_thresh: polygon confidence threshold. Polygons with scores lower than this threshold are filtered out.
        max_candidates: maximum number of proposed polygons.
        expand_ratio: controls by how much polygons need to be expanded to recover the original text shape
            (DBNet predicts shrunken text masks).
        output_polygon: output polygons or rectangles as the network's predictions.
        pred_name: heatmap's name used for polygons extraction.
        rescale_fields: name of fields to scale back to the shape of the original image.
    """

    def __init__(
        self,
        binary_thresh: float = 0.3,
        box_thresh: float = 0.7,
        max_candidates: int = 1000,
        expand_ratio: float = 1.5,
        box_type="quad",
        pred_name: str = "binary",
        rescale_fields: List[str] = ["polys"],
    ):
        super().__init__(box_type, rescale_fields)

        self._min_size = 3
        self._binary_thresh = binary_thresh
        self._box_thresh = box_thresh
        self._max_candidates = max_candidates
        self._expand_ratio = expand_ratio
        self._out_poly = box_type == "poly"
        self._name = pred_name
        self._names = {"binary": 0, "thresh": 1, "thresh_binary": 2}

    def _postprocess(
        self, pred: Union[Tensor, Tuple[Tensor], np.ndarray], **kwargs
    ) -> dict:
        """
        Postprocess network prediction to get text boxes on the transformed image space (which will be rescaled back to original image space in __call__ function)

        Args:
			pred (Union[Tensor, Tuple[Tensor], np.ndarray]): network prediction consists of
				binary: text region segmentation map, with shape (N, 1, H, W)
				thresh: [if exists] threshold prediction with shape (N, 1, H, W) (optional)
				thresh_binary: [if exists] binarized with threshold, (N, 1, H, W) (optional)

		Return:
            postprocessing result as a dict with keys:
                - polys (List[List[np.ndarray]): predicted polygons on the **transformed** (i.e. resized normally) image space, of shape (batch_size, num_polygons, num_points, 2). If `box_type` is 'quad', num_points=4.
                - scores (np.ndarray): confidence scores for the predicted polygons, shape (batch_size, num_polygons)
        """
        if isinstance(pred, tuple):
            pred = pred[self._names[self._name]]
        if isinstance(pred, Tensor):
            pred = pred.asnumpy()
        pred = pred.squeeze(1)

        segmentation = pred >= self._binary_thresh

        dest_size = np.array(pred.shape[:0:-1]) - 1

        polys, scores = [], []
        for pr, segm, size in zip(pred, segmentation, dest_size):
            sample_polys, sample_scores = self._extract_preds(pr, segm, size)
            polys.append(sample_polys)
            scores.append(sample_scores)

        output = {"polys": polys, "scores": scores}

        return output

    def _extract_preds(
        self, pred: np.ndarray, bitmap: np.ndarray, dest_size: np.ndarray
    ):
        outs = cv2.findContours(
            bitmap.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(outs) == 3:  # FIXME: update to OpenCV 4.x and delete this
            _, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        polys, scores = [], []
        for contour in contours[: self._max_candidates]:
            contour = contour.squeeze(1)
            score = self._calc_score(pred, bitmap, contour)
            if score < self._box_thresh:
                continue

            if self._out_poly:
                epsilon = 0.005 * cv2.arcLength(contour, closed=True)
                points = cv2.approxPolyDP(contour, epsilon, closed=True).squeeze(1)
                if points.shape[0] < 4:
                    continue
            else:
                points, min_side = self._fit_box(contour)
                if min_side < self._min_size:
                    continue

            poly = Polygon(points)
            poly = np.array(
                expand_poly(
                    points, distance=poly.area * self._expand_ratio / poly.length
                )
            )
            if self._out_poly and len(poly) > 1:
                continue
            poly = poly.reshape(-1, 2)

            _box, min_side = self._fit_box(poly)
            if min_side < self._min_size + 2:
                continue
            if not self._out_poly:
                poly = _box

            # TODO: an alternative solution to avoid calling self._fit_box twice:
            # box = Polygon(points)
            # box = np.array(expand_poly(points, distance=box.area * self._expand_ratio / box.length, joint_type=pyclipper.JT_MITER))
            # assert box.shape[0] == 4, print(f'box shape is {box.shape}')

            polys.append(np.clip(poly, 0, dest_size).astype(np.float32)) # keep float before rescaling
            scores.append(score)

        if self._out_poly:
            return polys, scores
        return np.array(polys), np.array(scores).astype(np.float32)

    @staticmethod
    def _fit_box(contour):
        """
        Finds a minimum rotated rectangle enclosing the contour.
        """
        # box = cv2.minAreaRect(contour)  # returns center of a rect, size, and angle
        # # TODO: does the starting point really matter?
        # points = np.roll(cv2.boxPoints(box), -1, axis=0)  # extract box points from a rotated rectangle
        # return points, min(box[1])
        # box = cv2.minAreaRect(contour)  # returns center of a rect, size, and angle
        # # TODO: does the starting point really matter?
        # points = np.roll(cv2.boxPoints(box), -1, axis=0)  # extract box points from a rotated rectangle
        # return points, min(box[1])

        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        # index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    @staticmethod
    def _calc_score(pred, mask, contour):
        """
        calculates score (mean value) of a prediction inside a given contour.
        """
        min_vals = np.clip(
            np.floor(np.min(contour, axis=0)), 0, np.array(pred.shape[::-1]) - 1
        ).astype(np.int32)
        max_vals = np.clip(
            np.ceil(np.max(contour, axis=0)), 0, np.array(pred.shape[::-1]) - 1
        ).astype(np.int32)

        return cv2.mean(
            pred[min_vals[1] : max_vals[1] + 1, min_vals[0] : max_vals[0] + 1],
            mask[min_vals[1] : max_vals[1] + 1, min_vals[0] : max_vals[0] + 1].astype(
                np.uint8
            ),
        )[0]
