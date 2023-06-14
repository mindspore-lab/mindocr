import argparse
import os
import sys

import numpy as np

import mindspore as ms
from mindspore import Tensor

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

from mindocr.metrics import build_metric  # noqa


def _rec_adapt_train_pred(content):
    return {"texts": [content]}


def _rec_adapt_train_label(content):
    return Tensor(np.array([content]))


def eval_rec_adapt_train(preds, labels, character_dict_path):
    metric_config = {
        "name": "RecMetric",
        "main_indicator": "acc",
        "character_dict_path": character_dict_path,
        "ignore_space": True,
        "print_flag": False,
    }
    metric = build_metric(metric_config)

    adapted_preds = {}
    for img_name, content in preds.items():
        adapted_pred = _rec_adapt_train_pred(content)
        adapted_preds[img_name] = adapted_pred

    adapted_labels = {}
    for img_name, content in labels.items():
        adapted_label = _rec_adapt_train_label(content)
        adapted_labels[img_name] = adapted_label

    for img_name, _ in adapted_labels.items():
        metric.update(adapted_preds[img_name], adapted_labels[img_name])

    eval_res = metric.eval()
    return eval_res


def read_content(filename):
    results = {}
    with open(filename, encoding="utf-8") as f:
        for line in f:
            name, content = line.split("\t", 1)
            results[name] = content.strip().replace('"', "")

    return results


if __name__ == "__main__":
    ms.set_context(device_target="CPU")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", required=True, type=str)
    parser.add_argument("--pred_path", required=True, type=str)
    parser.add_argument("--character_dict_path", required=False, default=None, type=str)
    args = parser.parse_args()

    gt_path = args.gt_path
    pred_path = args.pred_path
    character_dict_path = args.character_dict_path

    labels = read_content(gt_path)
    preds = read_content(pred_path)
    labels_keys = labels.keys()
    preds_keys = preds.keys()

    if set(labels_keys) != set(preds_keys):
        raise ValueError("The images in gt_path and pred_path must be the same.")

    print("----- Start adapted eval rec------")
    eval_res = eval_rec_adapt_train(preds, labels, character_dict_path)
    print(eval_res)
    print("----- End adapted eval rec------")
