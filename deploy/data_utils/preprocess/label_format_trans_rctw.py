import os
from tqdm import tqdm
import shutil
import numpy as np
from shapely.geometry import Polygon
import argparse


"""
ICDAR2017-RCTW

download link: https://rctw.vlrlab.net/dataset, Training images and annotations_v1.2(7.6G)

(1) mkdir images && mkdir labels

(2) unzip dataset file
images：cat train_images.zip.* > train_images.zip && unzip train_images.zip
labels: unzip train_gts.zip

(3) run 
python label_format_trans_rctw.py --src_labels_path=train_gts \
                                  --src_images_path=train_images \
                                  --tgt_labels_path=labels \
                                  --tgt_images_path=images
"""

def label_format_trans_rctw(src_labels_path, src_images_path, tgt_labels_path, tgt_images_path):
    for filename in tqdm(os.listdir(src_labels_path)):
        src_image_filename = filename.replace("txt", "jpg")
        tgt_image_filename = src_image_filename.replace("image", "gt")
        src_image_filename = os.path.join(src_images_path, src_image_filename)
        tgt_image_filename = os.path.join(tgt_images_path, tgt_image_filename)
        assert os.path.exists(src_image_filename)
        shutil.copyfile(src_image_filename, tgt_image_filename)
        src_label_filename = os.path.join(src_labels_path, filename)
        output = []
        with open(src_label_filename, encoding="utf-8-sig") as f:
            content = f.readlines()
            lines = [line.strip() for line in content]

            for line in lines:
                split_line = line.split(",", 9)
                if split_line[-2] == '1':
                    continue

                if "," in split_line[-1]:
                    text = text.replace(",", " ")

                points = [int(x) for x in split_line[:8]]
                polygon = Polygon(np.array(points).reshape(4, 2)).convex_hull
                if polygon.area == 0:
                    continue

                text = split_line[-1].strip("\"")
                output.append(','.join([*split_line[:8], text]))

        output = [line + '\n' for line in output]
        tgt_label_filename = os.path.join(tgt_labels_path, filename.replace("image", "gt_img"))
        with open(tgt_label_filename, 'w', encoding='utf-8') as f:
            f.writelines(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_labels_path', type=str, required=True)
    parser.add_argument('--src_images_path', type=str, required=True)
    parser.add_argument('--tgt_labels_path', type=str, required=True)
    parser.add_argument('--tgt_images_path', type=str, required=True)

    args = parser.parse_args()
    label_format_trans_rctw(args.src_labels_path, args.src_images_path, args.tgt_labels_path, args.tgt_images_path)