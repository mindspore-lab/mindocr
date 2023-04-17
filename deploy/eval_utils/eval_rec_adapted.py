import argparse
import json
import os
import sys

import numpy as np


__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))
from mindocr.metrics import build_metric
from mindspore import Tensor

def rec_adapt_train_pred(content):
    texts = []
    for con in content:
        line = {'texts': [con['transcription']],
                'conf': [1.0],
                'raw_chars': [[]]}
        texts.append(line)
    # if not boxes:   # TODO: texts is empty
    #     return None

    return texts

def rec_adapt_train_label(content):
    texts = []
    for con in content:
        line = Tensor(np.array([con['transcription']]))    # TODO: cannot get `gt_lens`, refer to rec_metrics.py
        texts.append(line)
    # if not boxes:     # TODO: texts is empty
    #     return None

    return texts

def eval_rec_adapt_train(preds, labels):
    metric_config = {"name": "RecMetric", "main_indicator": "acc", "character_dict_path": None,
                          "ignore_space": True, "print_flag": False}
    metric = build_metric(metric_config)

    adapted_preds = {}
    for img_name, content in preds.items():
        adapted_pred = rec_adapt_train_pred(content)
        adapted_preds[img_name] = adapted_pred

        # if adapted_pred:    # TODO: texts is not empty
        #     adapted_preds.append(adapted_pred)

    adapted_labels = {}
    for img_name, content in labels.items():
        adapted_label = rec_adapt_train_label(content)
        adapted_labels[img_name] = adapted_label

        # if adapted_label:   # TODO: texts is not empty
        #     adapted_labels.append(adapted_label)

    for img_name, label in adapted_labels.items():
        for i, lab in enumerate(label): # TODO: how to ensure the order of text within one image match in gt and preds
            # import pdb
            # pdb.set_trace()
            metric.update(adapted_preds[img_name][i], lab)

    eval_res = metric.eval()
    return eval_res

def read_content(filename):
    results = {}
    with open(filename, encoding='utf-8') as f:
        for line in f:
            name, content = line.split('\t', 2)
            results[name] = json.loads(content)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', required=True, type=str)
    parser.add_argument('--pred_path', required=True, type=str)
    parser.add_argument('--parallel_num', required=False, type=int, default=32)
    args = parser.parse_args()

    gt_path = args.gt_path
    pred_path = args.pred_path
    parallel_num = args.parallel_num

    labels = read_content(gt_path)
    preds = read_content(pred_path)
    labels_keys = labels.keys()
    preds_keys = preds.keys()

    if set(labels_keys) != set(preds_keys):
        raise ValueError(f"The images in gt_path and pred_path must be the same.")


    print('----- Start adapted eval rec------')
    eval_res = eval_rec_adapt_train(preds, labels)
    print(eval_res)
    print('----- End adapted eval rec------')
