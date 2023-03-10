import os
import json
import shutil
import argparse
from tqdm import tqdm
import numpy as np


"""
ICDAR2019-LSVT

download link: 
https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_0.tar.gz
https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_1.tar.gz
https://dataset-bj.cdn.bcebos.com/lsvt/train_full_labels.json

(1) mkdir images && mkdir labels

(2) unzip dataset file
tar -zvxf ./train_full_images_0.tar.gz
tar -zvxf ./train_full_images_1.tar.gz
mv train_full_images_0/* images
mv train_full_images_1/* images

(3) run
python label_format_trans_lsvt.py --label_json_path=train_full_labels.json --output_path=labels

"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_json_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    label_path = args.label_json_path

    save_path = args.output_path
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, mode=0o750, exist_ok=True)
    label_dict = {}
    with open(label_path, 'r') as f:
        label_dict.update(json.load(f))
    for file_name in tqdm(label_dict):
        write_list = []
        for info in label_dict[file_name]:
            rec_label = [info['transcription']]
            det_label = np.array(info['points']).reshape(-1).tolist()
            det_label = list(map(str, det_label))
            combine_label = ','.join(det_label + rec_label)
            write_list.append(combine_label)
        split_name = (file_name.split('.')[0]).split('_')
        save_name = split_name[0] + '_img_' + split_name[1]
        with open(os.path.join(save_path, save_name + '.txt'), 'w', encoding='utf8') as f:
            for item in write_list:
                f.write("%s\n" % item)
