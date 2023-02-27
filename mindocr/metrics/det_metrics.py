'''
Code adopted from paddle.

'''

__all__ = ['DetMetric']

from .eval_det_iou import DetectionIoUEvaluator


class DetMetric(object):
    def __init__(self, main_indicator='hmean', **kwargs):
        self.evaluator = DetectionIoUEvaluator()
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        '''
        Args:
            batch: a list produced by dataloaders.
               image: np.ndarray  of shape (N, C, H, W).
               ratio_list: np.ndarray  of shape(N,2)
               polygons: np.ndarray  of shape (N, K, 4, 2), the polygons of objective regions.
               ignore_tags: np.ndarray  of shape (N, K), indicates whether a region is ignorable or not.
            preds: a list of dict produced by post process
                points: np.ndarray of shape (N, K, 4, 2), the polygons of objective regions.
       '''
        gt_polyons_batch = batch[2]
        ignore_tags_batch = batch[3]
        for pred, gt_polyons, ignore_tags in zip(preds, gt_polyons_batch,
                                                 ignore_tags_batch):
            # prepare gt
            gt_info_list = [{
                'points': gt_polyon,
                'text': '',
                'ignore': ignore_tag
            } for gt_polyon, ignore_tag in zip(gt_polyons, ignore_tags)]
            # prepare det
            det_info_list = [{
                'points': det_polyon,
                'text': ''
            } for det_polyon in pred['points']]
            result = self.evaluator.evaluate_image(gt_info_list, det_info_list)
            self.results.append(result)

    def get_metric(self):
        """
        return metrics {
                 'precision': 0,
                 'recall': 0,
                 'hmean': 0
            }
        """

        metrics = self.evaluator.combine_results(self.results)
        self.reset()
        return metrics

    def reset(self):
        self.results = []  # clear results
