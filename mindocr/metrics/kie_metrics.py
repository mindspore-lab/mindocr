import json
import os
from glob import glob

import numpy as np
import seqeval.metrics
import sklearn

from mindspore import get_context, nn

from mindocr.utils.kie_utils import Synchronizer
from mindocr.utils.misc import AllReduce

__all__ = ["VQASerTokenMetric", "VQAReTokenMetric"]


class VQASerTokenMetric(nn.Metric):
    """
    Metric method for token classification.
    """

    def __init__(self, device_num: int = 1, **kwargs):
        super().__init__()
        self.clear()
        self.device_num = device_num
        self.synchronizer = None if device_num <= 1 else Synchronizer(device_num)
        self.metric_names = ["precision", "recall", "hmean"]
        if "save_dir" in kwargs:
            self.save_dir = kwargs["save_dir"]

    def update(self, output_batch, gt):
        preds, gt = output_batch
        self.pred_list.extend(preds)
        self.gt_list.extend(gt)

    def eval(self):
        gt_list = self.gt_list
        pred_list = self.pred_list

        if self.synchronizer:
            eval_dir = os.path.join(self.save_dir, "eval_tmp")
            os.makedirs(eval_dir, exist_ok=True)

            device_id = get_context("device_id")
            eval_path = os.path.join(eval_dir, f"eval_result_{device_id}.txt")
            with open(eval_path, "w") as fp:
                json.dump({"gt_list": gt_list, "pred_list": pred_list}, fp)
            self.synchronizer()

            eval_files = glob(eval_dir + "/*")
            gt_list = []
            pred_list = []
            for e_file in eval_files:
                with open(e_file, "r") as fp:
                    eval_info = json.load(fp)
                    gt_list += eval_info["gt_list"]
                    pred_list += eval_info["pred_list"]
        metrics = {
            "precision": seqeval.metrics.precision_score(gt_list, pred_list),
            "recall": seqeval.metrics.recall_score(gt_list, pred_list),
            "hmean": seqeval.metrics.f1_score(gt_list, pred_list),
        }
        return metrics

    def clear(self):
        self.pred_list = []
        self.gt_list = []


class VQAReTokenMetric(nn.Metric):
    """
    Metric method for Token RE task.
    """

    def __init__(self, device_num: int = 1, **kwargs):
        super().__init__()
        self.clear()
        self.device_num = device_num
        self.all_reduce = AllReduce(reduce="sum") if device_num > 1 else None
        self.metric_names = ["precision", "recall", "hmean"]

    def update(self, logits, labels):
        self.preds_list.extend(np.argmax(logits, axis=-1))
        self.labels_list.extend(labels[0].numpy())

    def eval(self):
        preds = np.concatenate(self.preds_list)
        labels = np.concatenate(self.labels_list)
        valid_indices = labels != -100
        valid_preds = preds[valid_indices]
        valid_labels = labels[valid_indices]

        precision = sklearn.metrics.precision_score(valid_labels, valid_preds)
        recall = sklearn.metrics.recall_score(valid_labels, valid_preds, average="binary")
        h_mean = sklearn.metrics.f1_score(valid_labels, valid_preds, average="binary")

        metrics = {
            "precision": precision,
            "recall": recall,
            "hmean": h_mean,
        }
        return metrics

    def clear(self):
        self.preds_list = []
        self.labels_list = []
