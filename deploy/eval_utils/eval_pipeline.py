import argparse
import json

import numpy as np
from joblib import Parallel, delayed
from shapely.geometry import Polygon
from tqdm import tqdm


def read_content(filename):
    results = {}
    with open(filename, encoding="utf-8") as f:
        for line in f:
            name, content = line.split("\t", 1)
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
            # only with valid word label proves the validity of item
            gt_text = gt_item["transcription"].replace(" ", "").lower()
            pred_text = pred_item["transcription"].replace(" ", "").lower()
            if gt_text == pred_text:
                matched_count += 1

    return matched_count


def each_recognition_eval(label_itmes, pred_items):
    label_itmes = [items for items in label_itmes if items["transcription"] not in ("###", "*")]

    correct_num, total_num = 0, len(label_itmes)

    for pred_item in pred_items:
        matched_num = process_words(label_itmes, pred_item)
        correct_num += matched_num

    return (correct_num, total_num)


def eval_rec(labels, preds, parallel_num):
    res = Parallel(n_jobs=parallel_num, backend="multiprocessing")(
        delayed(each_recognition_eval)(labels[key], preds[key]) for key in tqdm(labels.keys())
    )

    res = np.array(res)
    correct_num = sum(res[:, 0])
    total_num = sum(res[:, 1])
    acc = correct_num / total_num if total_num else 0
    result = {"acc:": acc, "correct_num:": correct_num, "total_num:": total_num}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", required=True, type=str)
    parser.add_argument("--pred_path", required=True, type=str)
    parser.add_argument("--parallel_num", required=False, type=int, default=32)
    args = parser.parse_args()

    gt_path = args.gt_path
    pred_path = args.pred_path
    parallel_num = args.parallel_num

    labels = read_content(gt_path)
    preds = read_content(pred_path)

    labels_keys = labels.keys()
    preds_keys = preds.keys()

    if set(labels_keys) != set(preds_keys):
        raise ValueError("The images in gt_path and pred_path must be the same.")

    result_rec = eval_rec(labels, preds, parallel_num)
    print(result_rec)
