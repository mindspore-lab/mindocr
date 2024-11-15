"""
Text classification inference

Example:
    $ python tools/infer/text/predict_cls.py  --image_dir {path_to_img} --cls_algorithm MV3
"""
import logging
import os
from typing import List

import numpy as np
from postprocess import Postprocessor
from preprocess import Preprocessor

import mindspore as ms
from mindspore import ops

from mindocr import build_model
from mindocr.utils.logger import set_logger
from mindocr.utils.visualize import show_imgs
from tools.infer.text.utils import get_ckpt_file, get_image_paths

algo_to_model_name = {
    "MV3": "cls_mobilenet_v3_small_100_model",
}
logger = logging.getLogger("mindocr")


class TextClassifier(object):
    """
    Infer model for text orientation classification
    """

    def __init__(self, args):
        self.batch_num = args.cls_batch_num
        self.batch_mode = args.cls_batch_mode
        logger.info(
            "classify in {} mode {}".format(
                "batch" if self.batch_mode else "serial",
                "batch_size: " + str(self.batch_num) if self.batch_mode else "",
            )
        )

        # build model for algorithm with pretrained weights or local checkpoint
        ckpt_dir = args.cls_model_dir
        if ckpt_dir is None:
            pretrained = True
            ckpt_load_path = None
        else:
            ckpt_load_path = get_ckpt_file(ckpt_dir)
            pretrained = False
        assert args.cls_algorithm in algo_to_model_name, (
            f"Invalid cls_algorithm {args.cls_algorithm}. "
            f"Supported classification algorithms are {list(algo_to_model_name.keys())}"
        )
        model_name = algo_to_model_name[args.cls_algorithm]

        amp_level = args.cls_amp_level
        if amp_level != "O0" and args.cls_algorithm == "mv3":
            logger.warning("The MV3 model supports only amp_level O0")
            amp_level = "O0"

        self.model = build_model(model_name, pretrained=pretrained, ckpt_load_path=ckpt_load_path, amp_level=amp_level)
        self.model.set_train(False)

        self.cast_pred_fp32 = amp_level != "O0"
        if self.cast_pred_fp32:
            self.cast = ops.Cast()
        logger.info(
            "Init classification model: {} --> {}. Model weights loaded from {}".format(
                args.cls_algorithm, model_name, "pretrained url" if pretrained else ckpt_load_path
            )
        )

        # build preprocess
        self.preprocess = Preprocessor(
            task="cls",
            algo=args.cls_algorithm,
        )

        # build postprocess
        self.postprocess = Postprocessor(task="cls", algo=args.cls_algorithm)

        self.vis_dir = args.draw_img_save_dir
        os.makedirs(self.vis_dir, exist_ok=True)

    def __call__(self, img_or_path_list: list, do_visualize: bool = False) -> List:
        """
        Run text classification serially for input images

        Args:
            img_or_path_list: list of str for img path or np.array for RGB image
            do_visualize: visualize preprocess and final result and save them

        Return:
            list of dict, each contains the follow keys for classification result.
            e.g. [{'angle': 180, 'score': 1.0}, {'angle': 0, 'score': 1.0}]
                - angle: text angle
                - score: prediction confidence
        """

        assert isinstance(
            img_or_path_list, list
        ), "Input for text classification must be list of images or image paths."
        logger.info(f"num images for cls: {len(img_or_path_list)}")

        if self.batch_mode:
            cls_res_all_crops = self.run_batchwise(img_or_path_list, do_visualize)
        else:
            cls_res_all_crops = []
            for i, img_or_path in enumerate(img_or_path_list):
                cls_res = self.run_single(img_or_path, i, do_visualize)
                cls_res_all_crops.append(cls_res)

        return cls_res_all_crops

    def run_batchwise(self, img_or_path_list: list, do_visualize=False):
        """
        Run text angle classification serially for input images

            Args:
            img_or_path_list: list of str for img path or np.array for RGB image
            do_visualize: visualize preprocess and final result and save them

            Return:
            cls_res: list of tuple, where each tuple is  (angle, score)
            - text angle classification result for each input image in order.
                where text is the predicted text string, scores is its confidence score.
                e.g. [(180, 0.9), (0, 1.0)]
        """

        cls_res = []
        num_imgs = len(img_or_path_list)

        for idx in range(0, num_imgs, self.batch_num):
            batch_begin = idx
            batch_end = min(idx + self.batch_num, num_imgs)
            logger.info(f"CLS img idx range: [{batch_begin}, {batch_end})")
            img_batch = []

            # preprocess
            for j in range(batch_begin, batch_end):
                data = self.preprocess(img_or_path_list[j])
                img_batch.append(data["image"])
                if do_visualize:
                    f_name = os.path.basename(data.get("img_path", f"crop_{j}.png")).rsplit(".", 1)[0]
                    show_imgs(
                        [data["image"]],
                        title=f_name + "_cls_preprocessed",
                        mean_rgb=[127.0, 127.0, 127.0],
                        std_rgb=[127.0, 127.0, 127.0],
                        is_chw=True,
                        show=False,
                        save_path=os.path.join(self.vis_dir, f_name + "_cls_preproc.png"),
                    )

            # infer
            img_batch = np.stack(img_batch) if len(img_batch) > 1 else np.expand_dims(img_batch[0], axis=0)

            net_pred = self.model(ms.Tensor(img_batch))
            if self.cast_pred_fp32:
                if isinstance(net_pred, (list, tuple)):
                    net_pred = [self.cast(p, ms.float32) for p in net_pred]
                else:
                    net_pred = self.cast(net_pred, ms.float32)

            # postprocess
            batch_res = self.postprocess(net_pred)
            cls_res.extend(list(zip(batch_res["angles"], batch_res["scores"])))

        return cls_res

    def run_single(self, img_or_path, crop_idx=0, do_visualize=True):
        """
        Text angle classification inference on a single image

        Args:
            img_or_path: str for image path or np.array for image RGB value

        Return:
            dict with keys:
                - angle: text angle
                - score: prediction confidence
        """

        # preprocess
        data = self.preprocess(img_or_path)

        # visualize preprocess result
        if do_visualize:
            f_name = os.path.basename(data.get("img_path", f"crop_{crop_idx}.png")).rsplit(".", 1)[0]
            show_imgs(
                [data["image"]],
                title=f_name + "_cls_preprocessed",
                mean_rgb=[127.0, 127.0, 127.0],
                std_rgb=[127.0, 127.0, 127.0],
                is_chw=True,
                show=False,
                save_path=os.path.join(self.vis_dir, f_name + "_cls_preproc.png"),
            )

        # infer
        input_np = data["image"]
        if len(input_np.shape) == 3:
            net_input = np.expand_dims(input_np, axis=0)

        net_pred = self.model(ms.Tensor(net_input))
        if self.cast_pred_fp32:
            if isinstance(net_pred, (list, tuple)):
                net_pred = [self.cast(p, ms.float32) for p in net_pred]
            else:
                net_pred = self.cast(net_pred, ms.float32)

        # postprocess
        cls_res_raw = self.postprocess(net_pred)
        cls_res = list(zip(cls_res_raw["angles"], cls_res_raw["scores"]))

        logger.info(f"Crop {crop_idx} cls result: {cls_res}")

        return cls_res


def save_cls_res(cls_res_all, img_paths, batch_mode, include_score=True, save_path="./cls_results.txt"):
    """
    Generate cls_results files that store the angle classification results.

    Args:
        cls_res_all: list of dict, each contains the follow keys for classification result.
        img_paths: list of str for img path
        batch_mode: whether to run classification inference in batch-mode
        include_score: whether to write prediction confidence
        save_path: file storage path

    Return:
        lines: The content of Angle information written to the document
    """

    lines = []
    for i, cls_res in enumerate(cls_res_all):
        if not batch_mode:
            cls_res = cls_res[0]
        if include_score:
            img_pred = os.path.basename(img_paths[i]) + "\t" + cls_res[0] + "\t" + str(cls_res[1]) + "\n"
        else:
            img_pred = os.path.basename(img_paths[i]) + "\t" + cls_res[0] + "\n"
        lines.append(img_pred)

    with open(save_path, "w", encoding="utf-8") as f_cls:
        f_cls.writelines(lines)
        f_cls.close()

    return lines


if __name__ == "__main__":
    from time import time

    from config import parse_args

    # parse args
    args = parse_args()
    set_logger(name="mindocr")
    save_dir = args.draw_img_save_dir
    img_paths = get_image_paths(args.image_dir)

    ms.set_context(mode=args.mode)

    # init detector
    text_classification = TextClassifier(args)

    # run for each image
    start = time()
    cls_res_all = text_classification(img_paths, do_visualize=False)
    t = time() - start

    # save all results in a txt file
    save_fp = os.path.join(save_dir, "cls_results.txt" if args.cls_batch_mode else "cls_results_serial.txt")
    save_cls_res(cls_res_all, img_paths, args.cls_batch_mode, save_path=save_fp)

    # log the result of Angle classification inference
    logger.info(f"All cls res: {cls_res_all}")
    logger.info(f"Done! Text angle classification results saved in {save_dir}")
    logger.info(f"Time cost: {t}, FPS: {len(img_paths) / t}")
