import argparse
import json
import os

import numpy as np
import yaml
from shapely.geometry import Polygon
from tqdm import tqdm

"""
ICDAR2019-ReCTS

download link: https://rrc.cvc.uab.es/?ch=12, Training Set

(1) mkdir images

(2) unzip dataset file
unzip ReCTS.zip && cd detection

(3) run
python label_format_trans_rects.py --src_labels_path=gt --tgt_labels_path=labels.txt
"""


def label_format_trans_rects(src_labels_path, tgt_labels_path):
    '''
    Format annotation to standard form for ReCTS dataset.
    '''
    save_dict = {}

    for filename in tqdm(os.listdir(src_labels_path)):
        if not filename.endswith(".json"):
            continue

        image_filename = os.path.splitext(filename)[0] + '.jpg'
        save_dict[image_filename] = []
        with open(os.path.join(src_labels_path, filename), encoding='utf-8') as f:
            content = yaml.safe_load(f)

        lines = content.get('lines', [])

        for line in lines:
            if line['ignore'] == 1:
                continue

            text = line['transcription']
            points = line['points']

            # check illegal polygon
            polygon = Polygon(np.array(points).reshape(4, 2)).convex_hull
            if polygon.area <= 0:
                continue

            save_dict[image_filename].append({"transcription": text, "points": points})

    with open(tgt_labels_path, 'w', encoding='utf-8') as f:
        if not save_dict:
            f.write('')
        for name, res in save_dict.items():
            content = name + '\t' + json.dumps(res, ensure_ascii=False) + "\n"
            f.write(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_labels_path', type=str, required=True,
                        help='Directory of the labels, such as rects/detection/gt.')
    parser.add_argument('--tgt_labels_path', type=str, default="labels.txt", required=False,
                        help='Path to save the converted annotation.')
    args = parser.parse_args()

    if not os.path.isdir(args.src_labels_path):
        raise ValueError(f"Please make sure that {args.src_labels_path} is a dir.")

    label_format_trans_rects(args.src_labels_path, args.tgt_labels_path)
