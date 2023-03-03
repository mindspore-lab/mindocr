'''
Code adopted from paddle.
TODO: overwrite

'''

from typing import List

import numpy as np
from mindspore import nn
from shapely.geometry import Polygon

__all__ = ['DetMetric']



def _get_intersect(pD, pG):
    return pD.intersection(pG).area


def _get_iou(pD, pG):
    return pD.intersection(pG).area / pD.union(pG).area


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DetectionIoUEvaluator:
    def __init__(self, min_iou=0.5, min_intersect=0.5):
        self._min_iou = min_iou
        self._min_intersect = min_intersect

    def evaluate_image(self, gt: List[dict], preds: List[np.ndarray]):
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

        pairs = []
        det_match = 0
        iou_mat = np.zeros([1, 1])
        if gt_polys and det_polys:
            iou_mat = np.zeros([len(gt_polys), len(det_polys)])
            det_rect_mat = np.zeros(len(det_polys), np.int8)

            for gt_idx in range(len(gt_polys)):
                for det_idx in range(len(det_polys)):
                    if det_rect_mat[det_idx] == 0:  # the match is not found yet
                        iou_mat[gt_idx, det_idx] = _get_iou(det_polys[det_idx], gt_polys[gt_idx])
                        if iou_mat[gt_idx, det_idx] > self._min_iou:
                            # Mark the visit arrays
                            det_rect_mat[det_idx] = 1
                            det_match += 1
                            pairs.append({'gt': gt_idx, 'det': det_idx})
                            break

        if not gt_polys:
            recall = 1.
            precision = 0. if det_polys else 1.
        else:
            recall = det_match / len(gt_polys)
            precision = det_match / len(det_polys) if det_polys else 0.
        hmean = 0. if (precision + recall) == 0 else \
                2. * precision * recall / (precision + recall)

        return {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'iou_mat': [] if len(det_polys) > 100 else iou_mat.tolist(),
            'gt_polys': gt_polys,
            'det_polys': det_polys,
            'gt_num': len(gt_polys),
            'det_num': len(det_polys),
            'gt_ignore': gt_ignore,
            'det_ignore': det_ignore,
            'det_matched': det_match
        }

    def combine_results(self, results):
        num_global_care_gt = 0
        num_global_care_det = 0
        matched_sum = 0
        for result in results:
            num_global_care_gt += result['gt_num']
            num_global_care_det += result['det_num']
            matched_sum += result['det_matched']

        method_recall = 0 if num_global_care_gt == 0 else float(
            matched_sum) / num_global_care_gt
        method_precision = 0 if num_global_care_det == 0 else float(
            matched_sum) / num_global_care_det
        methodHmean = 0 if method_recall + method_precision == 0 else 2 * method_recall * method_precision / \
                                                                      (method_recall + method_precision)
        method_metrics = {'precision': method_precision,
                          'recall': method_recall, 'hmean': methodHmean}
        return method_metrics


class QuadMetric:
    def __init__(self, is_output_polygon=False):
        self.is_output_polygon = is_output_polygon
        self.evaluator = DetectionIoUEvaluator()

    def measure(self, batch, output, box_thresh=0.7):
        '''
        batch: (image, polygons, ignore_tags)
            image: numpy array of shape (N, C, H, W).
            polys: numpy array of shape (N, K, 4, 2), the polygons of objective regions.
            dontcare: numpy array of shape (N, K), indicates whether a region is ignorable or not.
        output: (polygons, ...)
        '''
        gt_polys = batch['polys'].astype(np.float32)
        gt_ignore_info = batch['ignore']
        pred_polys = np.array(output[0])
        pred_scores = np.array(output[1])

        result = []
        for sample_id in range(len(gt_polys)):
            gt = [{'polys': gt_polys[sample_id][j], 'ignore': gt_ignore_info[sample_id][j]}
                  for j in range(len(gt_polys[sample_id]))]
            if self.is_output_polygon:
                pred = [sample for sample in pred_polys[sample_id]]    # TODO: why polygons are not filtered?
            else:
                pred = [pred_polys[sample_id][j].astype(np.int32)
                        for j in range(pred_polys[sample_id].shape[0]) if pred_scores[sample_id][j] >= box_thresh]
            result.append(self.evaluator.evaluate_image(gt, pred))
        return result


    def validate_measure(self, batch, output):
        return self.measure(batch, output, box_thresh=0.55)     # TODO: why is here a fixed threshold and different from the above?

    def gather_measure(self, raw_metrics):
        raw_metrics = [image_metrics for image_metrics in raw_metrics]

        result = self.evaluator.combine_results(raw_metrics)

        precision = AverageMeter()
        recall = AverageMeter()
        hmean = AverageMeter()

        precision.update(result['precision'], n=len(raw_metrics))
        recall.update(result['recall'], n=len(raw_metrics))
        hmean_score = 2 * precision.val * recall.val / (precision.val + recall.val + 1e-8)
        hmean.update(hmean_score)

        return {
            'precision': precision,
            'recall': recall,
            'hmean': hmean
        }

# TODO: improve the efficiency ? 
class DetMetric(nn.Metric):
    def __init__(self, use_iou_rotate=False, **kwargs):
        super().__init__()
        self.clear()

    def clear(self):
        self._metric = QuadMetric()
        self._raw_metrics = []

    def update(self, *inputs):
        '''
        compute metric on a batch of data

        Args: 
            inputs (tuple): contain two elements preds, gt
                    preds (dict): prediction output by postprocess, required keys: 
                        - polygons
                        - scores
                    gt (tuple): ground truth, order defined by output_keys in eval dataloader
        '''
        preds, gts = inputs
        pred_polys, scores = preds['polygons'], preds['scores']
        gt_polys, ignore_tags = gts # defined by out_keys in eval dataloader 
        
        if not isinstance(gt_polys, np.ndarray):
            gt_polys = gt_polys.asnumpy()
            ignore_tags = ignore_tags.asnumpy()

        gt = {'polys': gt_polys, 'ignore': ignore_tags}
        self._raw_metrics.extend(self._metric.validate_measure(gt, (pred_polys, scores)))

    def eval(self):
        '''
        Evaluate by aggregting results from batch update

        Returns: dict, average precision, recall, hmean of all samples 
            precision: precision,
            recall: recall,
            hmean: hmean
        '''
        measures = self._metric.gather_measure(self._raw_metrics)

        avg_measures = {k:v.avg for k, v in measures.items()}
        return avg_measures

if  __name__ == '__main__':
    m = DetMetric()

