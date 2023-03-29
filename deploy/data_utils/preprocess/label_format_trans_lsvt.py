import argparse
import json
import os

"""
ICDAR2019-LSVT

download link: 
https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_0.tar.gz
https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_1.tar.gz
https://dataset-bj.cdn.bcebos.com/lsvt/train_full_labels.json

(1) mkdir images

(2) unzip dataset file
tar -zvxf ./train_full_images_0.tar.gz
tar -zvxf ./train_full_images_1.tar.gz
mv train_full_images_0/* images
mv train_full_images_1/* images

(3) run
python label_format_trans_lsvt.py --src_labels_path=train_full_labels.json --tgt_labels_path=labels.txt

"""


def label_format_trans_lsvt(src_labels_path, tgt_labels_path):
    '''
    Format annotation to standard form for LSVT dataset.
    '''
    with open(src_labels_path) as f:
        samples = json.load(f)

    save_dict = {}

    for filename, boxes in samples.items():
        filename = filename + ".jpg"
        save_dict[filename] = []
        for box in boxes:
            if box["illegibility"]:
                continue
            save_dict[filename].append({"transcription": box['transcription'], "points": box['points']})

    with open(tgt_labels_path, 'w', encoding='utf-8') as f:
        if not save_dict:
            f.write('')
        for name, res in save_dict.items():
            content = name + '\t' + json.dumps(res, ensure_ascii=False) + "\n"
            f.write(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_labels_path', type=str, required=True,
                        help='Path of the labels json file, such as lsvt/train_full_labels.json.')
    parser.add_argument('--tgt_labels_path', type=str, default="labels.txt", required=False,
                        help='Path to save the converted annotation.')
    args = parser.parse_args()

    if not os.path.isfile(args.src_labels_path):
        raise ValueError(f"Please make sure that {args.src_labels_path} is a file.")

    label_format_trans_lsvt(args.src_labels_path, args.tgt_labels_path)
