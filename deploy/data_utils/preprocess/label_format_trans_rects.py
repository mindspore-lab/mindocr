import os
from tqdm import tqdm
import shutil
import numpy as np
from shapely.geometry import Polygon
import argparse
import yaml

"""
ICDAR2019-ReCTS

download link: https://rrc.cvc.uab.es/?ch=12
If download failed, you can refer to: https://github.com/WenmuZhou/OCR_DataSet，ReCTS.zip

(1) mkdir images && mkdir labels

(2) unzip dataset file
unzip ReCTS.zip && cd detection

(3) run
python label_format_trans_rects.py --src_labels_path=gt \
                                   --src_images_path=img \
                                   --tgt_labels_path=labels \
                                   --tgt_images_path=images
"""

def label_format_trans_rects(src_labels_path, src_images_path, tgt_labels_path, tgt_images_path):
    for idx, filename in tqdm(enumerate(os.listdir(src_labels_path)), total=len(os.listdir(src_labels_path))):
        if not filename.endswith(".json"):
            continue

        name = os.path.splitext(filename)[0]

        src_image_filename = name + '.jpg'
        tgt_image_filename = 'gt_' + str(idx) + '.jpg'
        src_image_filename = os.path.join(src_images_path, src_image_filename)
        tgt_image_filename = os.path.join(tgt_images_path, tgt_image_filename)
        assert os.path.exists(src_image_filename)
        shutil.copyfile(src_image_filename, tgt_image_filename)

        src_label_filename = os.path.join(src_labels_path, filename)
        tgt_label_filename = 'gt_img_' + str(idx) + '.txt'
        tgt_label_filename = os.path.join(tgt_labels_path, tgt_label_filename)

        with open(src_label_filename, encoding='utf-8') as f:
            content = yaml.safe_load(f)

        lines = content.get('lines', [])
        output = []
        for line in lines:
            if line['ignore'] == 1:
                continue

            text = line['transcription']
            points = line['points']

            if "," in text:
                text = text.replace(",", " ")

            polygon = Polygon(np.array(points).reshape(4, 2)).convex_hull
            if polygon.area == 0:
                continue

            box = [str(x) for x in points]
            output.append(','.join([*box, text]))

        output = [line + '\n' for line in output]
        with open(tgt_label_filename, 'w', encoding='utf-8') as f:
            f.writelines(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_labels_path', type=str, required=True)
    parser.add_argument('--src_images_path', type=str, required=True)
    parser.add_argument('--tgt_labels_path', type=str, required=True)
    parser.add_argument('--tgt_images_path', type=str, required=True)

    args = parser.parse_args()
    label_format_trans_rects(args.src_labels_path, args.src_images_path, args.tgt_labels_path, args.tgt_images_path)