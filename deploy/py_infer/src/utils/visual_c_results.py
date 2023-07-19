import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm
from visual_utils import vis_bbox_text


def img_write(path: str, img: np.ndarray):
    filename = os.path.abspath(path)
    cv2.imencode(os.path.splitext(filename)[1], img)[1].tofile(filename)


def vis_results(prediction_result, vis_pipeline_save_dir, img_folder):
    img_files = os.listdir(img_folder)
    img_dict = {}
    font_path = os.path.abspath("../../../../docs/fonts/simfang.ttf")
    for img_name in img_files:
        img = cv2.imread(os.path.join(img_folder, img_name))  # BGR format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_dict[img_name] = img

    for each_pred in tqdm(prediction_result):
        file_name, prediction = each_pred.split("\t")
        basename = os.path.basename(file_name)
        save_file = os.path.join(vis_pipeline_save_dir, os.path.splitext(basename)[0])
        prediction = eval(prediction)
        box_list = [np.array(x["points"]).reshape(-1, 2) for x in prediction]
        text_list = [x["transcription"] for x in prediction]
        box_text = vis_bbox_text(img_dict[file_name], box_list, text_list, font_path=font_path)
        img_write(save_file + ".jpg", box_text)


def read_prediction(prediction_folder):
    with open(prediction_folder, "r", encoding="utf-8") as f:
        prediction = f.readlines()
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_folder", required=True, type=str)
    parser.add_argument("--pred_dir", required=True, type=str)
    parser.add_argument("--vis_dir", required=True, type=str)
    args = parser.parse_args()

    prediction = read_prediction(args.pred_dir)
    vis_results(prediction, args.vis_dir, args.img_folder)
