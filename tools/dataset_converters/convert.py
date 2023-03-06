"""
Code adopted from https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/gen_label.py
"""
from pathlib import Path
import argparse
import json


def gen_rec_label(input_path, out_label):
    with open(out_label, 'w') as outf:
        with open(input_path, 'r') as f:
            for line in f:
                # , may occur in text
                sep_index = line.find(',')
                img_path = line[:sep_index].strip().replace('\ufeff', '')
                label = line[sep_index + 1:].strip().replace("\"", "")
                abs_img_path =
                outf.write(img_path + '\t' + label + '\n')


def gen_det_label(image_dir, label_dir, out_label):
    label_dir = Path(label_dir)
    with open(out_label, 'w') as out_file:
        for img_path in Path(image_dir).iterdir():
            label_path = label_dir / f'gt_{img_path.stem}.txt'
            label = []
            with open(label_path, 'r', encoding='utf-8-sig') as f:
                for line in f.readlines():
                    tmp = line.strip("\n\r").replace("\xef\xbb\xbf", "").split(',')
                    points = tmp[:8]
                    s = []
                    for i in range(0, len(points), 2):
                        b = points[i:i + 2]
                        b = [int(t) for t in b]
                        s.append(b)
                    result = {"transcription": tmp[8], "points": s}
                    label.append(result)

            out_file.write(str(img_path) + '\t' + json.dumps(label, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        default="rec",
        help='Generate rec_label or det_label, can be set rec or det')
    parser.add_argument(
        '--root_path',
        type=str,
        default=".",
        help='The root directory of images. Only takes effect when task=det')
    parser.add_argument(
        '--input_path',
        type=str,
        default=".",
        help='Input_label or input path to be converted')
    parser.add_argument(
        '--output_label',
        type=str,
        default="out_label.txt",
        help='Output file name')

    args = parser.parse_args()
    if args.task == "rec":
        print("Generate rec label")
        gen_rec_label(args.input_path, args.output_label)
    elif args.task == "det":
        gen_det_label(args.root_path, args.input_path, args.output_label)
