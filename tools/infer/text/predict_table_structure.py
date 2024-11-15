"""
Infer table structure from images.

Example:
    $ python tools/infer/text/predict_table_structure.py --image_dir {path_to_img} --table_algorithm TABLE_MASTER
"""
import logging
import os
import sys
import time
from typing import Dict, Union

import numpy as np
from config import parse_args
from postprocess import Postprocessor
from preprocess import Preprocessor
from utils import get_ckpt_file, get_image_paths

from mindspore import Tensor

from mindocr.models import build_model
from mindocr.utils.logger import set_logger
from mindocr.utils.visualize import draw_boxes, show_imgs

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))


algo_to_model_name = {
    "TABLE_MASTER": "table_master",
}
logger = logging.getLogger("mindocr")


class StructureAnalyzer:
    """
    Model inference class for table structure analysis.
    Example:
        >>> args = parse_args()
        >>> analyzer = StructureAnalyzer(args)
        >>> img_path = "path/to/image.jpg"
        >>> (structure_str_list, bbox_list), elapsed_time = analyzer(img_path)
    """

    def __init__(self, args):
        ckpt_dir = args.table_model_dir
        if ckpt_dir is None:
            pretrained = True
            ckpt_load_path = None
        else:
            ckpt_load_path = get_ckpt_file(ckpt_dir)
            pretrained = False
        if args.table_algorithm not in algo_to_model_name:
            raise ValueError(
                f"Invalid table algorithm {args.table_algorithm}. "
                f"Supported table algorithms are {list(algo_to_model_name.keys())}"
            )
        model_name = algo_to_model_name[args.table_algorithm]

        self.model = build_model(
            model_name, pretrained=pretrained, ckpt_load_path=ckpt_load_path, amp_level=args.table_amp_level
        )
        self.model.set_train(False)
        self.preprocess = Preprocessor(task="table", table_max_len=args.table_max_len)
        self.postprocess = Postprocessor(task="table", table_char_dict_path=args.table_char_dict_path)
        self.vis_dir = args.draw_img_save_dir
        os.makedirs(self.vis_dir, exist_ok=True)

    def __call__(
        self,
        img_or_path: Union[str, np.ndarray, Dict],
        do_visualize: bool = True,
    ):
        """
        Perform model inference.
        Args:
            img_or_path (Union[str, np.ndarray, Dict]): Input image or image path.
            do_visualize (bool): Whether to visualize the result.
        Returns:
            Structure string list, bounding box list, and elapsed time.
        """
        time_profile = {}
        start_time = time.time()
        data = self.preprocess(img_or_path)
        input_np = data["image"]
        if len(input_np.shape) == 3:
            input_np = Tensor(np.expand_dims(input_np, axis=0))

        net_pred = self.model(input_np)
        shape_list = np.expand_dims(data["shape"], axis=0)
        post_result = self.postprocess(net_pred, labels=[shape_list])
        structure_str_list = post_result["structure_batch_list"][0][0]
        structure_str_list = ["<html>", "<body>", "<table>"] + structure_str_list + ["</table>", "</body>", "</html>"]
        bbox_list = post_result["bbox_batch_list"][0]
        elapse = time.time() - start_time
        time_profile["structure"] = elapse

        if do_visualize:
            vst = time.time()
            img_name = os.path.basename(data.get("img_path", "input.png")).rsplit(".", 1)[0]
            save_path = os.path.join(self.vis_dir, img_name + "_structure.png")
            structure_vis = draw_boxes(img_or_path, bbox_list, draw_type="rectangle")
            show_imgs([structure_vis], show=False, save_path=save_path)
            time_profile["vis"] = time.time() - vst
        return (structure_str_list, bbox_list), time_profile


def main():
    args = parse_args()
    set_logger(name="mindocr")
    analyzer = StructureAnalyzer(args)
    img_paths = get_image_paths(args.image_dir)
    for i, img_path in enumerate(img_paths):
        logger.info(f"Inferring {i+1}/{len(img_paths)}: {img_path}")
        _ = analyzer(img_path, do_visualize=True)
    logger.info(f"Done! All structure results are saved to {args.draw_img_save_dir}")


if __name__ == "__main__":
    main()
