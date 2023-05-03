import argparse
import json
import os
import sys
import warnings

import numpy as np


__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))
from mindocr.metrics import build_metric
from mindspore import Tensor

def _det_adapt_train_pred(content):
    boxes = []
    for con in content:
        if isinstance(con, dict):
            box = np.array(con['points']).reshape((4, 2))
        else:
            box = np.array(con).reshape((4, 2))
        boxes.append(box)

    boxes = np.array(boxes)

    conf_score = np.ones([len(boxes)])   # TODO: Hard code condidence score to be 1, which is not true.
    return [(boxes, conf_score)]

def _det_adapt_train_label(content):
    boxes = []
    for con in content:
        boxes.append(np.array(con['points']).reshape((4, 2)))

    ignored_tag = []
    for con in content:
        text = con['transcription']
        if text in ("###", "*"):
            ignored_tag.append(True)
        else:
            ignored_tag.append(False)
    ignored_tag = Tensor(np.array(ignored_tag).reshape(1, -1))

    boxes = Tensor(np.expand_dims(boxes, axis=0))
    return [boxes, ignored_tag]

def eval_det_adapt_train(preds, labels):
    metric_config = {"name": "DetMetric", "main_indicator": "acc", "print_flag": False}
    metric = build_metric(metric_config)

    adapted_preds = {}
    for img_name, content in preds.items():
        if not content:  # content is empty
            continue
        adapted_pred = _det_adapt_train_pred(content)
        adapted_preds[img_name] = adapted_pred

    adapted_labels = {}
    for img_name, content in labels.items():
        if not content:  # content is empty
            continue
        adapted_label = _det_adapt_train_label(content)
        adapted_labels[img_name] = adapted_label

    if len(adapted_preds) != len(adapted_labels):
        print(f"WARNING: The len of adapted_preds ({len(adapted_preds)}) is not equal to the len of adapted_labels ({len(adapted_labels)})."
                      "Some contents are empty in pred or labels files.")

    for img_name, label in adapted_labels.items():
        pred = adapted_preds.get(img_name, None)
        if pred:   # TODO: ignore the empty preds img, but it should be a valid but wrong pred results
            metric.update(pred, label)

    eval_res = metric.eval()
    return eval_res

def read_content(filename):
    results = {}
    with open(filename, encoding='utf-8') as f:
        for line in f:
            name, content = line.split('\t', 1)
            results[name] = json.loads(content)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', required=True, type=str)
    parser.add_argument('--pred_path', required=True, type=str)
    args = parser.parse_args()

    gt_path = args.gt_path
    pred_path = args.pred_path

    labels = read_content(gt_path)
    preds = read_content(pred_path)
    labels_keys = labels.keys()
    preds_keys = preds.keys()

    if set(labels_keys) != set(preds_keys):
        raise ValueError(f"The images in gt_path and pred_path must be the same.")

    print('----- Start adapted eval det------')
    eval_res = eval_det_adapt_train(preds, labels)
    print(eval_res)
    print('----- End adapted eval det------')
