from typing import List

import numpy as np
import mindspore as ms
from mindspore import nn, ms_function
import mindspore.ops as ops
from mindspore import  Tensor
from mindspore.communication import get_group_size
from shapely.geometry import Polygon
from sklearn.metrics import recall_score, precision_score, f1_score

__all__ = ['DetMetric']


def _get_intersect(pd, pg):
    return pd.intersection(pg).area


def _get_iou(pd, pg):
    return pd.intersection(pg).area / pd.union(pg).area


class DetectionIoUEvaluator:
    def __init__(self, min_iou=0.5, min_intersect=0.5):
        self._min_iou = min_iou
        self._min_intersect = min_intersect

    def __call__(self, gt: List[dict], preds: List[np.ndarray]):
        # filter invalid groundtruth polygons and split them into useful and ignored
        gt_polys, gt_ignore = [], []
        for sample in gt:
            poly = Polygon(sample['polys'])
            if poly.is_valid and poly.is_simple:
                if not sample['ignore']:
                    gt_polys.append(poly)
                else:
                    gt_ignore.append(poly)

        # repeat the same step for the predicted polygons
        det_polys, det_ignore = [], []
        for pred in preds:
            poly = Polygon(pred)
            if poly.is_valid and poly.is_simple:
                poly_area = poly.area
                if gt_ignore and poly_area > 0:
                    for ignore_poly in gt_ignore:
                        intersect_area = _get_intersect(ignore_poly, poly)
                        precision = intersect_area / poly_area
                        # If precision enough, append as ignored detection
                        if precision > self._min_intersect:
                            det_ignore.append(poly)
                            break
                    else:
                        det_polys.append(poly)
                else:
                    det_polys.append(poly)

        det_labels = [0] * len(gt_polys)
        if det_polys:
            iou_mat = np.zeros([len(gt_polys), len(det_polys)])
            det_rect_mat = np.zeros(len(det_polys), np.int8)

            for det_idx in range(len(det_polys)):
                if det_rect_mat[det_idx] == 0:  # the match is not found yet
                    for gt_idx in range(len(gt_polys)):
                        iou_mat[gt_idx, det_idx] = _get_iou(det_polys[det_idx], gt_polys[gt_idx])
                        if iou_mat[gt_idx, det_idx] > self._min_iou:
                            # Mark the visit arrays
                            det_rect_mat[det_idx] = 1
                            det_labels[gt_idx] = 1
                            break
                    else:
                        det_labels.append(1)

        gt_labels = [1] * len(gt_polys) + [0] * (len(det_labels) - len(gt_polys))
        return gt_labels, det_labels


class DetMetric(nn.Metric):
    def __init__(self, device_num=1, **kwargs):
        super().__init__()
        self._evaluator = DetectionIoUEvaluator()
        self._gt_labels, self._det_labels = [], []
        self.device_num = device_num
        self.all_reduce = None if device_num==1 else ops.AllReduce()
        self.metric_names = ['recall', 'precision', 'f-score']

    def clear(self):
        self._gt_labels, self._det_labels = [], []

    def update(self, *inputs):
        """
        compute metric on a batch of data

        Args:
            inputs (tuple): contain two elements preds, gt
                    preds (list[tuple]): text detection prediction as a list of tuple (polygon, confidence),
                        where polygon is in shape [num_boxes, 4, 2], confidence is in shape [num_boxes]
                    gts (tuple): ground truth - (polygons, ignore_tags), where polygons are in shape [num_images, num_boxes, 4, 2],
                        ignore_tags are in shape [num_images, num_boxes], which can be defined by output_columns in yaml
        """
        preds, gts = inputs
        polys, ignore = gts[0].asnumpy().astype(np.float32), gts[1].asnumpy()

        for sample_id in range(len(polys)):
            gt = [{'polys': poly, 'ignore': ig} for poly, ig in zip(polys[sample_id], ignore[sample_id])]
            gt_label, det_label = self._evaluator(gt, preds[sample_id][0])
            self._gt_labels.append(gt_label)
            self._det_labels.append(det_label)

    @ms_function
    def all_reduce_fun(self, x):
        res = self.all_reduce(x)
        return res

    def cal_matrix(self, det_lst, gt_lst):
        tp = np.sum((gt_lst == 1) * (det_lst == 1))
        fn = np.sum((gt_lst == 1) * (det_lst == 0))
        fp = np.sum((gt_lst == 0) * (det_lst == 1))
        return tp, fp, fn



    def eval(self):
        """
        Evaluate by aggregating results from batch update

        Returns: dict, average precision, recall, f1-score of all samples
            precision: precision,
            recall: recall,
            f-score: f-score
        """
        # flatten predictions and labels into 1D-array
        self._det_labels = np.array([l for label in self._det_labels for l in label])
        self._gt_labels = np.array([l for label in self._gt_labels for l in label])

        tp, fp, fn = self.cal_matrix(self._det_labels, self._gt_labels)
        if self.all_reduce:
            tp = float(self.all_reduce_fun(Tensor(tp, ms.float32)).asnumpy())
            fp = float(self.all_reduce_fun(Tensor(fp, ms.float32)).asnumpy())
            fn = float(self.all_reduce_fun(Tensor(fn, ms.float32)).asnumpy())

        recall = _safe_divide(tp, (tp + fn))
        precision = _safe_divide(tp, (tp + fp))
        f_score = _safe_divide(2 * recall * precision, (recall + precision))
        return {
            'recall': recall,
            'precision': precision,
            'f-score': f_score
        }


def _safe_divide(numerator, denominator, val_if_zero_divide=0.):
    if denominator == 0:
        return val_if_zero_divide
    else:
        return numerator / denominator
