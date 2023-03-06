"""
Code adopted from paddle.
TODO: overwrite
"""
from typing import List

import numpy as np
from mindspore import nn
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


class QuadMetric:
    def __init__(self, is_output_polygon=False):
        self.is_output_polygon = is_output_polygon
        self.evaluator = DetectionIoUEvaluator()

    def __call__(self, batch, output, box_thresh=0.7):
        """
        batch: (image, polygons, ignore_tags)
            image: numpy array of shape (N, C, H, W).
            polys: numpy array of shape (N, K, 4, 2), the polygons of objective regions.
            ignore: numpy array of shape (N, K), indicates whether a region is ignorable or not.
        output: (polygons, ...)
        """
        gt_polys = batch['polys'].astype(np.float32)
        gt_ignore_info = batch['ignore']
        pred_polys = np.array(output[0])
        pred_scores = np.array(output[1])

        gt_labels, det_labels = [], []
        for sample_id in range(len(gt_polys)):
            gt = [{'polys': gt_polys[sample_id][j], 'ignore': gt_ignore_info[sample_id][j]}
                  for j in range(len(gt_polys[sample_id]))]
            if self.is_output_polygon:
                pred = [sample for sample in pred_polys[sample_id]]     # TODO: why are polygons not filtered?
            else:
                pred = [pred_polys[sample_id][j].astype(np.int32)
                        for j in range(pred_polys[sample_id].shape[0]) if pred_scores[sample_id][j] >= box_thresh]

            gt_label, det_label = self.evaluator(gt, pred)
            gt_labels.append(gt_label)
            det_labels.append(det_label)
        return gt_labels, det_labels

    def validate_measure(self, batch, output):
        return self(batch, output, box_thresh=0.55)  # TODO: why is here a fixed threshold and different from the above?


# TODO: improve the efficiency ?
class DetMetric(nn.Metric):
    def __init__(self, **kwargs):
        super().__init__()
        self.clear()

    def clear(self):
        self._metric = QuadMetric()
        self._gt_labels, self._det_labels = [], []

    def update(self, *inputs):
        """
        compute metric on a batch of data

        Args:
            inputs (tuple): contain two elements preds, gt
                    preds (dict): prediction output by postprocess, required keys:
                        - polygons
                        - scores
                    gt (tuple): ground truth, order defined by output_keys in eval dataloader
        """
        preds, gts = inputs
        polys, ignore = gts
        boxes, scores = preds['polygons'], preds['scores']
        gt = {'polys': polys.asnumpy(), 'ignore': ignore.asnumpy()}

        gt_labels, det_labels = self._metric.validate_measure(gt, (boxes, scores))
        self._gt_labels.extend(gt_labels)
        self._det_labels.extend(det_labels)

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
        return {
            'recall': recall_score(self._gt_labels, self._det_labels),
            'precision': precision_score(self._gt_labels, self._det_labels),
            'f-score': f1_score(self._gt_labels, self._det_labels)
        }


if __name__ == '__main__':
    m = DetMetric()
