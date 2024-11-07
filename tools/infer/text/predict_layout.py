"""
Infer layout from images using yolov8 model.

Example:
    $ python tools/infer/text/predict_layout.py --image_dir {path_to_img} --layout_algorithm YOLOv8
"""
import json
import logging
import os
from typing import Dict, List

import cv2
import numpy as np
from postprocess import Postprocessor
from preprocess import Preprocessor
from utils import get_ckpt_file, get_image_paths

import mindspore as ms

from mindocr import build_model
from mindocr.utils.logger import set_logger

algo_to_model_name = {
    "YOLOv8": "yolov8",
}
logger = logging.getLogger("mindocr")


class LayoutAnalyzer(object):
    """
    Infer model for layout analysis
    """

    def __init__(self, args):
        self.img_dir = os.path.dirname(args.image_dir)
        self.vis_dir = args.draw_img_save_dir

        # build model for algorithm with pretrained weights or local checkpoint
        ckpt_dir = args.layout_model_dir
        if ckpt_dir is None:
            pretrained = True
            ckpt_load_path = None
            pretrained_backbone = False
        else:
            ckpt_load_path = get_ckpt_file(ckpt_dir)
            pretrained = False
            pretrained_backbone = False
        if args.layout_algorithm not in algo_to_model_name:
            raise ValueError(
                f"Invalid layout algorithm {args.layout_algorithm}. "
                f"Supported layout algorithms are {list(algo_to_model_name.keys())}"
            )
        model_name = algo_to_model_name[args.layout_algorithm]
        self.model = build_model(
            model_name,
            pretrained=pretrained,
            pretrained_backbone=pretrained_backbone,
            ckpt_load_path=ckpt_load_path,
            amp_level=args.layout_amp_level,
        )
        self.model.set_train(False)

        self.preprocess = Preprocessor(task="layout")
        self.postprocess = Postprocessor(task="layout")

    def __call__(self, img_path: str, do_visualize: bool = False) -> List:
        """
            Args:
        img_path: image path
        do_visualize: visualize preprocess and final result and save them

            Return:
        results (list): layout results lists of dictionary:
                        - category_id (int): category id of the layout element(text, title, list, table, figure)
                        - bbox (list): bounding box of the layout element
                        - score (float): confidence score of the layout element
        """
        data = self._load_image(img_path)
        # preprocess
        data = self.preprocess(data)
        self.hw_ori = data["hw_ori"]
        self.hw_scale = data["hw_scale"]
        self.pad = data["pad"]

        data = np.ascontiguousarray(data["image"])

        # prepare network input
        net_input = ms.Tensor([data], ms.float32)
        self.img_shape = net_input.shape

        # infer
        preds = self.model(net_input)

        # postprocess
        results = self.postprocess(
            preds, img_shape=[self.img_shape], meta_info=([0], [self.hw_ori], [self.hw_scale], [self.pad])
        )

        if do_visualize:
            img_name = os.path.basename(img_path).rsplit(".", 1)[0]
            visualize_layout(img_path, results, save_path=os.path.join(self.vis_dir, img_name + "_layout_result.png"))

        return results

    def _load_image(self, img_path: str) -> Dict:
        """
        Load image from path
        """
        image = cv2.imread(img_path)
        h_ori, w_ori = image.shape[:2]  # orig hw
        hw_ori = np.array([h_ori, w_ori])
        target_size = 800
        r = target_size / max(h_ori, w_ori)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            image = cv2.resize(image, (int(w_ori * r), int(h_ori * r)), interpolation=interp)

        data = {"image": image, "raw_img_shape": hw_ori, "target_size": target_size}
        return data


def visualize_layout(image_path, results, conf_thres=0.8, save_path: str = ""):
    """
    Visualize layout analysis results
    """
    from matplotlib import pyplot as plt
    from PIL import Image

    img = Image.open(image_path)
    img_cv = cv2.imread(image_path)

    fig, ax = plt.subplots()
    ax.imshow(img)

    category_dict = {1: "text", 2: "title", 3: "list", 4: "table", 5: "figure"}
    color_dict = {1: (255, 0, 0), 2: (0, 0, 255), 3: (0, 255, 0), 4: (0, 255, 255), 5: (255, 0, 255)}

    for item in results:
        category_id = item["category_id"]
        bbox = item["bbox"]
        score = item["score"]

        if score < conf_thres:
            continue

        left, bottom, w, h = bbox
        right = left + w
        top = bottom + h

        cv2.rectangle(img_cv, (int(left), int(bottom)), (int(right), int(top)), color_dict[category_id], 2)

        label = "{} {:.2f}".format(category_dict[category_id], score)
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        cv2.rectangle(
            img_cv,
            (int(left), int(bottom - label_size[1] - base_line)),
            (int(left + label_size[0]), int(bottom)),
            color_dict[category_id],
            cv2.FILLED,
        )
        cv2.putText(
            img_cv, label, (int(left), int(bottom - base_line)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    if save_path:
        cv2.imwrite(save_path, img_cv)
    else:
        plt.axis("off")
        plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        plt.show()


def save_layout_res(layout_res: List, img_path: str, save_dir: str):
    """
    Save layout analysis results in a txt file
    """
    lines = []
    img_name = os.path.basename(img_path).rsplit(".", 1)[0]
    save_path = os.path.join(save_dir, img_name + "_layout_result.txt")
    for i, res in enumerate(layout_res):
        img_pred = str(json.dumps(res)) + "\n"
        lines.append(img_pred)

    with open(save_path, "w") as f:
        f.writelines(lines)
        f.close()


if __name__ == "__main__":
    from config import parse_args

    # parse args
    args = parse_args()
    set_logger(name="mindocr")
    save_dir = args.draw_img_save_dir
    img_paths = get_image_paths(args.image_dir)

    ms.set_context(mode=args.mode)

    # init layout analyzer
    layout_analyzer = LayoutAnalyzer(args)

    save_dir, _ = os.path.splitext(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # run for each image
    for i, img_path in enumerate(img_paths):
        logger.info(f"Infering [{i+1}/{len(img_paths)}]: {img_path}")

        layout_res = layout_analyzer(img_path, do_visualize=args.visualize_output)
        logger.info(f"Num analyze layout boxes: {len(layout_res)}")

        # save all results in a txt file
        save_layout_res(layout_res, img_path, save_dir=os.path.join(save_dir))

    logger.info(f"Done! layout analyze result saved in {save_dir}")
