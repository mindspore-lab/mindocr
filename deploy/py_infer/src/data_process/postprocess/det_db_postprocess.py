import logging
import os
import sys
from typing import List, Tuple, Union

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

import mindspore as ms
from mindspore import Tensor

# add mindocr root path, and import postprocess from mindocr
mindocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
sys.path.insert(0, mindocr_path)

from mindocr.postprocess import det_base_postprocess  # noqa

_logger = logging.getLogger(__name__)
__all__ = ["DBV4Postprocess"]


class DBV4Postprocess(det_base_postprocess.DetBasePostprocess):
    """
    The post process for DBNet, adapted to paddleocrV4.
    """

    def __init__(
        self,
        binary_thresh: float = 0.3,
        box_thresh: float = 0.7,
        max_candidates: int = 1000,
        expand_ratio: float = 1.5,
        box_type: str = "quad",
        pred_name: str = "binary",
        rescale_fields: List[str] = ["polys"],
        if_merge_longedge_bbox: bool = True,
        merge_inter_area_thres: int = 300,
        merge_ratio: float = 1.3,
        merge_angle_theta: float = 10,
        if_sort_bbox: bool = True,
        sort_bbox_y_delta: int = 10,
    ):
        super().__init__(rescale_fields, box_type)

        self._min_size = 3
        self._binary_thresh = binary_thresh
        self._box_thresh = box_thresh
        self._max_candidates = max_candidates
        self._expand_ratio = expand_ratio
        self._if_merge_longedge_bbox = if_merge_longedge_bbox
        self._merge_inter_area_thres = merge_inter_area_thres
        self._merge_ratio = merge_ratio
        self._merge_angle_theta = merge_angle_theta
        self._if_sort_bbox = if_sort_bbox
        self._sort_bbox_y_delta = sort_bbox_y_delta
        self._out_poly = box_type == "poly"
        self._name = pred_name
        self._names = {"binary": 0, "thresh": 1, "thresh_binary": 2}

    def __call__(
        self,
        pred: Union[ms.Tensor, Tuple[ms.Tensor], np.ndarray],
        shape_list: Union[np.ndarray, ms.Tensor] = None,
        **kwargs,
    ) -> dict:
        if isinstance(shape_list, Tensor):
            shape_list = shape_list.asnumpy()
        if shape_list is not None:
            assert shape_list.shape[0] and shape_list.shape[1] == 4, (
                "The shape of each item in shape_list must be 4: [raw_img_h, raw_img_w, scale_h, scale_w]. "
                f"But got shape_list of shape {shape_list.shape}"
            )
        else:
            _logger.warning(
                "`shape_list` is None in postprocessing. Cannot rescale the prediction result to original "
                "image space, which can lead to inaccurate evaluation. You may add `shape_list` to `output_columns` "
                "list under eval section in yaml config file, or directly parse `shape_list` to postprocess callable "
                "function."
            )
            self.warned = True
        result = self._postprocess(pred, shape_list=shape_list)
        src_w, src_h = shape_list[0, 1], shape_list[0, 0]
        polys = self.filter_tag_det_res(result["polys"][0], [src_h, src_w])
        if self._if_merge_longedge_bbox:
            try:
                polys = longedge_bbox_merge(
                    polys, self._merge_inter_area_thres, self._merge_ratio, self._merge_angle_theta
                )
            except Exception as e:
                _logger.warning(f"long edge bbox merge failed: {e}")
        if self._if_sort_bbox:
            polys = sorted_boxes(polys, self._sort_bbox_y_delta)
        result["polys"][0] = polys
        result["scores"].clear()
        return result

    def _postprocess(self, pred: Union[Tensor, Tuple[Tensor], np.ndarray], **kwargs) -> dict:
        """
        Postprocess network prediction to get text boxes on the transformed image space (which will be rescaled back to
        original image space in __call__ function)

        Args:
            pred (Union[Tensor, Tuple[Tensor], np.ndarray]): network prediction consists of
                binary: text region segmentation map, with shape (N, 1, H, W)
                thresh: [if exists] threshold prediction with shape (N, 1, H, W) (optional)
                thresh_binary: [if exists] binarized with threshold, (N, 1, H, W) (optional)

        Returns:
            postprocessing result as a dict with keys:
                - polys (List[np.ndarray]): predicted polygons on the **transformed** (i.e. resized normally) image
                space, of shape (batch_size, num_polygons, num_points, 2). If `box_type` is 'quad', num_points=4.
                - scores (np.ndarray): confidence scores for the predicted polygons, shape (batch_size, num_polygons)
        """
        if isinstance(pred, tuple):
            pred = pred[self._names[self._name]]
        if isinstance(pred, Tensor):
            pred = pred.asnumpy()
        if len(pred.shape) == 4 and pred.shape[1] != 1:  # pred shape (N, 3, H, W)
            pred = pred[:, 0, :, :]

        if len(pred.shape) == 4:  # handle pred shape: (N, H, W) skip
            pred = pred.squeeze(1)

        segmentation = pred >= self._binary_thresh
        polys, scores = [], []
        src_w, src_h = kwargs["shape_list"][0, 1], kwargs["shape_list"][0, 0]
        for pr, segm in zip(pred, segmentation):
            poly, score = self.boxes_from_bitmap(pr, segm, src_w, src_h)
            polys.append(poly)
            scores.append(score)
        return {"polys": polys, "scores": scores}

    def unclip(self, box, _expand_ratio):
        poly = Polygon(box)
        distance = poly.area * _expand_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
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

    def box_score_fast(self, bitmap, _box):
        """
        box_score_fast: use bbox mean score as the mean score
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        """
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        """

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            contours = outs[1]
        elif len(outs) == 2:
            contours = outs[0]

        num_contours = min(len(contours), self._max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self._min_size:
                continue
            points = np.array(points)

            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self._box_thresh > score:
                continue

            box = self.unclip(points, self._expand_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self._min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

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

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points


def longedge_bbox_merge(boxes, merge_inter_area_thres=300, merge_ratio=1.3, merge_angle_theta=10):
    """
    Merge long-edge bboxes according the following rule:
      - inter area larger than `merge_inter_area_thres`
      - delta of long edge slope of minimum outer rectangle larger than `merge_angle_theta`
      - short edge of merged boxes smaller than `merge_ratio` times short edge of boxes
    args:
        boxes(array): boxes to be merge, shape: (N, 4, 2). N: Number of bboxes
    return:
        merged boxes(array): merged boxes, shape: (N2, 4, 2). N2: Number of merged bboxes
    """
    ori_boxes = [box.tolist() for box in boxes]
    ori_poly = [Polygon(box) for box in ori_boxes]
    minrec_poly = [poly.minimum_rotated_rectangle for poly in ori_poly]

    merge_list = []
    merge_minrec = []
    check_merge = False

    while not check_merge or len(merge_list) > 0:
        merge_list.clear()
        merge_minrec.clear()

        for i in range(len(ori_boxes)):
            for j in range(i + 1, len(ori_boxes)):
                # inter area judgement
                inter_area = ori_poly[i].intersection(ori_poly[j]).area
                uij = ori_poly[i].union(ori_poly[j])
                if inter_area < merge_inter_area_thres:
                    continue
                minrec_i_theta, minrec_i_short_len = 0, 0
                minrec_i_xs, minrec_i_ys = minrec_poly[i].exterior.coords.xy
                minrec_i_edge1_len = np.sqrt(
                    (minrec_i_xs[1] - minrec_i_xs[0]) ** 2 + (minrec_i_ys[1] - minrec_i_ys[0]) ** 2
                )
                minrec_i_edge1_theta = np.arctan(
                    (minrec_i_ys[1] - minrec_i_ys[0]) / (minrec_i_xs[1] - minrec_i_xs[0] + 1e-5)
                )
                minrec_i_edge2_len = np.sqrt(
                    (minrec_i_xs[2] - minrec_i_xs[1]) ** 2 + (minrec_i_ys[2] - minrec_i_ys[1]) ** 2
                )
                minrec_i_edge2_theta = np.arctan(
                    (minrec_i_ys[2] - minrec_i_ys[1]) / (minrec_i_xs[2] - minrec_i_xs[1] + 1e-5)
                )

                if minrec_i_edge2_len > minrec_i_edge1_len:
                    minrec_i_theta = minrec_i_edge2_theta
                    minrec_i_short_len = minrec_i_edge1_len
                else:
                    minrec_i_theta = minrec_i_edge1_theta
                    minrec_i_short_len = minrec_i_edge2_len

                minrec_j_theta, minrec_j_short_len = 0, 0
                minrec_j_xs, minrec_j_ys = minrec_poly[j].exterior.coords.xy
                minrec_j_edge1_len = np.sqrt(
                    (minrec_j_xs[1] - minrec_j_xs[0]) ** 2 + (minrec_j_ys[1] - minrec_j_ys[0]) ** 2
                )
                minrec_j_edge1_theta = np.arctan(
                    (minrec_j_ys[1] - minrec_j_ys[0]) / (minrec_j_xs[1] - minrec_j_xs[0] + 1e-5)
                )
                minrec_j_edge2_len = np.sqrt(
                    (minrec_j_xs[2] - minrec_j_xs[1]) ** 2 + (minrec_j_ys[2] - minrec_j_ys[1]) ** 2
                )
                minrec_j_edge2_theta = np.arctan(
                    (minrec_j_ys[2] - minrec_j_ys[1]) / (minrec_j_xs[2] - minrec_j_xs[1] + 1e-5)
                )

                if minrec_j_edge2_len > minrec_j_edge1_len:
                    minrec_j_theta = minrec_j_edge2_theta
                    minrec_j_short_len = minrec_j_edge1_len
                else:
                    minrec_j_theta = minrec_j_edge1_theta
                    minrec_j_short_len = minrec_j_edge2_len

                # slope judgement
                if np.abs(minrec_j_theta - minrec_i_theta) > merge_angle_theta / 180 * np.pi:
                    continue

                # short edge judgement
                minrec_u = uij.minimum_rotated_rectangle
                minrec_u_xs, minrec_u_ys = minrec_u.exterior.coords.xy
                minrec_u_edge1_len = np.sqrt(
                    (minrec_u_xs[1] - minrec_u_xs[0]) ** 2 + (minrec_u_ys[1] - minrec_u_ys[0]) ** 2
                )
                minrec_u_edge2_len = np.sqrt(
                    (minrec_u_xs[2] - minrec_u_xs[1]) ** 2 + (minrec_u_ys[2] - minrec_u_ys[1]) ** 2
                )
                minrec_u_short_len = min(minrec_u_edge1_len, minrec_u_edge2_len)
                if minrec_u_short_len > merge_ratio * max(minrec_i_short_len, minrec_j_short_len):
                    continue

                merge_list.append([i, j])
                merge_minrec.append(minrec_u)

        if len(merge_minrec) > 0:
            ori_boxes = [ori_boxes[i] for i in range(len(ori_boxes)) if i not in merge_list[0]]
            ori_poly = [ori_poly[i] for i in range(len(ori_poly)) if i not in merge_list[0]]
            minrec_poly = [minrec_poly[i] for i in range(len(minrec_poly)) if i not in merge_list[0]]

            poly = merge_minrec[0]
            xs, ys = poly.exterior.coords.xy
            xs = xs.tolist()
            ys = ys.tolist()

            index = np.argsort(np.linalg.norm(np.array([xs[:-1], ys[:-1]]).T, ord=2, axis=1))[0]
            ori_boxes.append(
                [
                    [xs[index % 4], ys[index % 4]],
                    [xs[(index + 1) % 4], ys[(index + 1) % 4]],
                    [xs[(index + 2) % 4], ys[(index + 2) % 4]],
                    [xs[(index + 3) % 4], ys[(index + 3) % 4]],
                ]
            )

            ori_poly.append(Polygon(ori_boxes[-1]))
            minrec_poly.append(ori_poly[-1].minimum_rotated_rectangle)

        check_merge = True
    return np.array(ori_boxes)


def sorted_boxes(dt_boxes, sort_bbox_y_delta):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
        sort_bbox_y_delta:further sort boxes whose dy smaller than sort_bbox_y_delta
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = len(dt_boxes)
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < sort_bbox_y_delta and (
                _boxes[j + 1][0][0] < _boxes[j][0][0]
            ):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes
