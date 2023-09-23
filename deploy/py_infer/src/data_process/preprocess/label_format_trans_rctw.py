import argparse
import json
import os

import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm

"""
ICDAR2017-RCTW

download link: https://rctw.vlrlab.net/dataset, Training images and annotations_v1.2(7.6G)

(1) mkdir images

(2) unzip dataset file
images：cat train_images.zip.* > train_images.zip && unzip train_images.zip
labels: unzip train_gts.zip

(3) run 
python label_format_trans_rctw.py --src_labels_path=train_gts --tgt_labels_path=labels.txt
"""


def label_format_trans_rctw(src_labels_path, tgt_labels_path):
    '''
    Format annotation to standard form for RCTW-17 dataset.
    '''
    save_dict = {}

    for filename in tqdm(os.listdir(src_labels_path)):
        if not filename.endswith(".txt"):
            continue

        image_filename = filename.replace("txt", "jpg")
        src_label_filename = os.path.join(src_labels_path, filename)
        save_dict[image_filename] = []

        with open(src_label_filename, encoding="utf-8-sig") as f:
            content = f.readlines()
            lines = [line.strip() for line in content]

            for line in lines:
                split_line = line.split(",", 9)
                if split_line[-2] == '1':
                    continue

                # check illegal polygon
                points = [int(x) for x in split_line[:8]]
                polygon = Polygon(np.array(points).reshape(4, 2)).convex_hull
                if polygon.area <= 0:
                    continue

                text = split_line[-1].strip("\"")

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
                        help='Directory of the labels, such as rctw/train_gts.')
    parser.add_argument('--tgt_labels_path', type=str, default="labels.txt", required=False,
                        help='Path to save the converted annotation.')
    args = parser.parse_args()

    if not os.path.isdir(args.src_labels_path):
        raise ValueError(f"Please make sure that {args.src_labels_path} is a dir.")

    label_format_trans_rctw(args.src_labels_path, args.tgt_labels_path)
