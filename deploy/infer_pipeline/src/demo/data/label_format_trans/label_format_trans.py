# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
Description: ICDAR-2019 label format trans to ICDAR-2015 label format
Author: MindX SDK
Create: 2022
History: NA
"""

import os
import json
import shutil
import argparse
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_json_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    label_path = args.label_json_path

    save_path = os.path.join(args.output_path, 'labels')
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
