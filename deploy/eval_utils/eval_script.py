import argparse
import json
import os
import sys

import numpy as np
from joblib import Parallel, delayed
from shapely.geometry import Polygon
from tqdm import tqdm

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))
from mindocr.metrics import build_metric
from mindspore import Tensor

def read_content_adapt_train_pred(content):
    boxes = []
    for con in content:
        boxes.append(np.array(con['points']).reshape((4, 2)))

    if not boxes:   # TODO: boxes is empty
        return None
    boxes = np.array(boxes)

    conf_score = np.ones([len(boxes)])   # TODO: Hard code condidence score to be 1, which is not true.
    return [(boxes, conf_score)]

def read_content_adapt_train_label(content):
    boxes = []
    for con in content:
        boxes.append(np.array(con['points']).reshape((4, 2)))

    if not boxes:     # TODO: boxes is empty
        return None
    ignored_tag = Tensor(np.expand_dims(np.array([False] * len(boxes)), axis=0))
    boxes = Tensor(np.expand_dims(boxes, axis=0))
    return [boxes, ignored_tag]

def eval_det_adapt_train(preds, labels):
    # metric_config = {"name": "RecMetric", "main_indicator": "acc", "character_dict_path": None,
    #                       "ignore_space": True, "print_flag": False}
    metric_config = {"name": "DetMetric", "main_indicator": "acc",
                     "ignore_space": True, "print_flag": False}
    metric = build_metric(metric_config)

    adapted_preds = []
    for img_name, content in preds.items():
        adapted_pred = read_content_adapt_train_pred(content)
        if adapted_pred:    # TODO: boxes is not empty
            adapted_preds.append(adapted_pred)

    adapted_labels = []
    for img_name, content in labels.items():
        adapted_label = read_content_adapt_train_label(content)
        if adapted_label:   # TODO: boxes is not empty
            adapted_labels.append(adapted_label)

    for pred, label in zip(adapted_preds, adapted_labels):
        metric.update(pred, label)

    eval_res = metric.eval()
    return eval_res

def read_content(filename):
    results = {}
    with open(filename, encoding='utf-8') as f:
        for line in f:
            name, content = line.split('\t', 2)
            results[name] = json.loads(content)
    return results


def box_match_count(gt_polys, pred_poly, thresh=0.5):
    valid_count = 0
    for gt_poly in gt_polys:
        if not gt_poly.intersects(pred_poly):
            continue

        inter = gt_poly.intersection(pred_poly).area
        union = gt_poly.union(pred_poly).area

        if union > 0 and inter / union > thresh:
            valid_count += 1

    return valid_count


def process_words(label_itmes, pred_item, thresh=0.5):
    pred = np.array(pred_item["points"]).reshape((-1, 2))
    pred_poly = Polygon(pred)
    if not pred_poly.is_valid:
        return 0

    matched_count = 0
    for gt_item in label_itmes:
        gt = np.array(gt_item["points"]).reshape((-1, 2))
        gt_poly = Polygon(gt)
        if not gt_poly.is_valid:
            return 0

        if not gt_poly.intersects(pred_poly):
            continue

        inter = gt_poly.intersection(pred_poly).area
        ratio = 0
        if gt_poly.area:
            ratio = inter / gt_poly.area

        if ratio > thresh and gt_item["transcription"]:
            ## only with valid word label proves the validity of item
            gt_text = gt_item["transcription"].replace(" ", "")
            pred_text = pred_item["transcription"].replace(" ", "")
            if gt_text in pred_text:
                matched_count += 1

    return matched_count


def each_detection_eval(label_items, pred_items):
    matched = 0

    gt_polys = []
    for item in label_items:
        gt = np.array(item["points"]).reshape((-1, 2))
        gt_poly = Polygon(gt)
        if not gt_poly.is_valid or not gt_poly.is_simple:
            continue
        gt_polys.append(gt_poly)

    for item in pred_items:
        pred = np.array(item["points"]).reshape((-1, 2))
        pred_poly = Polygon(pred)
        if not pred_poly.is_valid or not pred_poly.is_simple:
            continue
        matched += box_match_count(gt_polys, pred_poly)

    result = {
        "matched": matched,
        "gt_num": len(gt_polys),
        "det_num": len(pred_items)
    }
    return result


def each_recognition_eval(label_itmes, pred_items):
    correct_num, total_num = 0, len(label_itmes)

    for pred_item in pred_items:
        matched_num = process_words(label_itmes, pred_item)
        correct_num += matched_num

    return (correct_num, total_num)


def eval_det(labels, preds, parallel_num):
    res = Parallel(n_jobs=parallel_num, backend="multiprocessing")(delayed(each_detection_eval)(
        labels[key], preds[key]) for key in tqdm(labels.keys()))

    matched_num = 0
    gt_num = 0
    det_num = 0
    for result in res:
        matched_num += result['matched']
        gt_num += result['gt_num']
        det_num += result['det_num']

    precision = 0 if not det_num else matched_num / det_num
    recall = 0 if not gt_num else matched_num / gt_num
    hmean = 0 if not precision + recall else 2 * precision * recall / (precision + recall)
    result = {
        "precision:": precision,
        "recall:": recall,
        "Hmean:": hmean,
        "matched:": matched_num,
        "det_num": det_num,
        "gt_num": gt_num
    }
    return result


def eval_rec(labels, preds, parallel_num):
    res = Parallel(n_jobs=parallel_num, backend="multiprocessing")(delayed(each_recognition_eval)(
        labels[key], preds[key]) for key in tqdm(labels.keys()))

    res = np.array(res)
    correct_num = sum(res[:, 0])
    total_num = sum(res[:, 1])
    acc = correct_num / total_num if total_num else 0
    result = {
        "acc:": acc,
        "correct_num:": correct_num,
        "total_num:": total_num
    }
    return result


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

    result_det = eval_det(labels, preds, parallel_num)
    print(result_det)

    result_rec = eval_rec(labels, preds, parallel_num)
    print(result_rec)

    print('----- Start adapted eval------')
    eval_res = eval_det_adapt_train(preds, labels)
    print(eval_res)
    print('----- End adapted eval------')
