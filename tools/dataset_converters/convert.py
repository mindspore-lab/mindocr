"""
Script to convert data annotation format for ocr model training

Example:
>>> python tools/dataset_converters/convert.py \
        --dataset_name  ic15 \
        --task det \
        --image_dir /path/to/ic15/det/train/ch4_training_images \
        --label_dir /path/to/ic15/det/train/ch4_training_localization_transcription_gt

>>> python tools/dataset_converters/convert.py \
        --dataset_name  ic15 \
        --task rec \
        --label_dir /path/to/ic15/rec/ch4_training_word_images_gt
"""

import argparse
import os

from ctw1500 import CTW1500_Converter
from ic15 import IC15_Converter
from mlt2017 import MLT2017_Converter
from svt import SVT_Converter
from syntext150k import SYNTEXT150K_Converter
from synthtext import SYNTHTEXT_Converter
from td500 import TD500_Converter
from totaltext import TOTALTEXT_Converter

supported_datasets = ["ic15", "totaltext", "mlt2017", "syntext150k", "svt", "td500", "ctw1500"]


def convert(dataset_name, task, image_dir, label_path, output_path=None, path_mode="relative"):
    """
    Args:
      image_dir: path to the images
      label_path: path to the annotation, support folder path or file path
      output_path: path to save the converted annotation. If None, the file will be saved as '{task}_gt.txt' along with
          `label_path`
    """
    if dataset_name in supported_datasets:
        if not output_path:
            root_dir = "/".join(label_path.split("/")[:-1])
            dir_name = os.path.basename(image_dir)
            output_path = os.path.join(root_dir, f"{dir_name}_{task}_gt.txt")
        assert path_mode in ["relative", "abs"], f"Invalid mode: {path_mode}"

        class_name = dataset_name.upper() + "_Converter"
        cvt = eval(class_name)(path_mode)
        cvt.convert(task, image_dir, label_path, output_path)
        print(f"Conversion complete.\nResult saved in {output_path}")

    else:
        raise ValueError(f"{dataset_name} is not supported for conversion, supported datasets are {supported_datasets}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset_name",
        type=str,
        default="ic15",
        help=f"Name of the dataset to convert. Valid choices: {supported_datasets}",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default="det",
        help="Target task, text detection or recognition, valid choices: det, rec",
    )
    parser.add_argument(
        "-i", "--image_dir", type=str, default="./ic15/det/images/", help="Directory to the images of the dataset"
    )
    parser.add_argument(
        "-l",
        "--label_dir",
        type=str,
        default="./ic15/det/annotation/",
        help="Directory of the labels (if many), or path to the label file (if one) of the dataset",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="",
        help="Path to save the converted annotation. If None, it will be saved as {task}_gt.txt along with label_dir",
    )
    parser.add_argument(
        "--path_mode",
        type=str,
        default="relative",
        help="If abs, the image path in the output annotation file will be an absolute path. If relative, "
        "it will be a relative path related to the image dir ",
    )

    args = parser.parse_args()

    convert(args.dataset_name, args.task, args.image_dir, args.label_dir, args.output_path)
