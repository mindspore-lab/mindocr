"""
Text detection inference

Example:
    $ python tools/infer/text/predict_layout.py  --image_dir {path_to_img}
                                                 --layout_algorithm LAYOUTLMV3
                                                 --config {config_path}
"""

import json
import logging
import os
import sys
from typing import List

import numpy as np
from config import parse_args
from postprocess import Postprocessor
from preprocess import Preprocessor
from utils import get_image_paths
import yaml
from addict import Dict

import mindspore as ms

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))

from mindocr import build_model
from mindocr.utils.logger import set_logger
from mindocr.utils.visualize import draw_boxes_for_layout, show_img

# map algorithm name to model name (which can be checked by `mindocr.list_models()`)
# NOTE: Modify it to add new model for inference.
algo_to_model_name = {
    "LAYOUTLMV3": "layoutlmv3"
}
logger = logging.getLogger("mindocr")


class LayoutDetector(object):
    def __init__(self, args):
        # build model for algorithm with pretrained weights or local checkpoint
        if args.config is None:
            raise ValueError(f"No config")
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        self.cfg = Dict(cfg)

        assert args.layout_algorithm in algo_to_model_name, (
            f"Invalid layout_algorithm {args.det_algorithm}. "
            f"Supported layout detection algorithms are {list(algo_to_model_name.keys())}"
        )
        model_name = algo_to_model_name[args.layout_algorithm]

        self.model = build_model(self.cfg.model, ckpt_load_path=self.cfg.predict.ckpt_load_path)
        self.model.set_train(False)
        logger.info(
            "Init detection model: {} --> {}. Model weights loaded from {}".format(
                args.layout_algorithm, model_name, self.cfg.predict.ckpt_load_path
            )
        )

        # build preprocess and postprocess
        self.preprocess = Preprocessor(
            task="layout",
            algo=args.layout_algorithm,
        )

        self.postprocess = Postprocessor(task="layout", algo=args.layout_algorithm)

        self.vis_dir = args.draw_img_save_dir
        os.makedirs(self.vis_dir, exist_ok=True)


    def __call__(self, img_or_path, do_visualize=True):

        # preprocess
        data = self.preprocess(img_or_path)
        fn = os.path.basename(data.get("img_path", "input.png")).rsplit(".", 1)[0]

        # infer
        input_np = data["image"]
        if len(input_np.shape) == 3:
            net_input = np.expand_dims(input_np, axis=0)
        hw_ori = data["raw_img_shape"]
        hw_scale = data["hw_scale"]
        input = [ms.Tensor(net_input), ms.Tensor(np.array([hw_ori])), ms.Tensor(np.array([hw_scale]))]
        net_output = self.model(*input)

        # postprocess
        kwargs = {}
        meta_info = [[img_or_path], [hw_ori], [hw_scale], None]
        kwargs["meta_info"] = meta_info
        kwargs["img_shape"] = hw_ori
        layout_res = self.postprocess(net_output, data, **kwargs)

        if do_visualize:
            det_vis = draw_boxes_for_layout(img_path, layout_res, self.cfg.predict.category_dict, self.cfg.predict.color_dict)
            show_img(
                det_vis, show=False, is_bgr_img=False, title=fn + "_layout_res", save_path=os.path.join(self.vis_dir, fn + "_layout_res.png")
            )

        return layout_res, data


def save_layout_res(layout_res_all: List[dict],save_path="./layout_results.txt"):
    fw = open(save_path, "w")
    for i, layout_res in enumerate(layout_res_all):
        fw.write(json.dumps(layout_res) + "\n")
    fw.close()


if __name__ == "__main__":
    # parse args
    args = parse_args()
    set_logger(name="mindocr")
    save_dir = args.draw_img_save_dir
    img_paths = get_image_paths(args.image_dir)
    # uncomment it to quick test the infer FPS
    # img_paths = img_paths[:15]

    ms.set_context(mode=args.mode)

    # init detector
    layout_detect = LayoutDetector(args)

    # run for each image
    det_layout_all = []
    for i, img_path in enumerate(img_paths):
        logger.info(f"\nInfering [{i+1}/{len(img_paths)}]: {img_path}")
        layout_res, _ = layout_detect(img_path, do_visualize=True)
        det_layout_all.append(layout_res)
        logger.info(f"Num detected text boxes: {len(layout_res)}")

    # save all results in a txt file
    save_layout_res(det_layout_all, save_path=os.path.join(save_dir, "layout_results.txt"))

    logger.info(f"Done! layout detection results saved in {save_dir}")
