from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

__all__ = ["YOLOv8Metric"]


class YOLOv8Metric(object):
    """Compute the mean average precision."""

    def __init__(self, annotations_path, device_num=1, **kwargs):
        self.annotations_path = annotations_path
        self.anno = COCO(annotations_path)  # init annotations api
        self.metric_names = ["map"]
        self.result_dicts = list()

    def update(self, preds, gt):
        self.result_dicts.extend(preds)

    def eval(self):
        pred = self.anno.loadRes(self.result_dicts)  # init predictions api
        coco_eval = COCOeval(self.anno, pred, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        map_result = coco_eval.stats[0]
        return {"map": map_result}

    def clear(self):
        self.result_dicts = list()
