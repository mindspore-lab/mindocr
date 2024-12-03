"""
Text detection and recognition inference

Example:
    $ python tools/infer/text/predict_system.py --image_dir {path_to_img_file} --det_algorithm DB++ \
      --rec_algorithm CRNN
    $ python tools/infer/text/predict_system.py --image_dir {path_to_img_dir} --det_algorithm DB++ \
      --rec_algorithm CRNN_CH
"""

import json
import logging
import os
import sys
from time import time
from typing import List, Union

import cv2
import numpy as np
from config import parse_args
from postprocess import Postprocessor
from predict_det import TextDetector
from predict_rec import TextRecognizer
from preprocess import Preprocessor
from utils import crop_text_region, get_image_paths, img_rotate

import mindspore as ms
from mindspore import ops

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))

from mindocr import build_model
from mindocr.utils.logger import set_logger
from mindocr.utils.visualize import visualize  # noqa
from tools.infer.text.utils import get_ckpt_file

logger = logging.getLogger("mindocr")


class TextClassifier(object):
    """
    Infer model for text orientation classification
    Example:
        >>> args = parse_args()
        >>> text_classification = TextClassifier(args)
        >>> img_path = "path/to/image.jpg"
        >>> cls_res_all = text_classification(image_path)
    """

    def __init__(self, args):
        algo_to_model_name = {
            "M3": "cls_mobilenet_v3_small_100_model",
        }
        self.batch_num = args.cls_batch_num
        logger.info("classify in {} mode {}".format("batch", "batch_size: " + str(self.batch_num)))

        # build model for algorithm with pretrained weights or local checkpoint
        ckpt_dir = args.cls_model_dir
        if ckpt_dir is None:
            pretrained = True
            ckpt_load_path = None
        else:
            ckpt_load_path = get_ckpt_file(ckpt_dir)
            pretrained = False

        assert args.cls_algorithm in algo_to_model_name, (
            f"Invalid cls_algorithm: {args.cls_algorithm}. "
            f"Supported classification algorithms are {list(algo_to_model_name.keys())}"
        )
        model_name = algo_to_model_name[args.cls_algorithm]

        amp_level = args.cls_amp_level
        if amp_level != "O0" and args.cls_algorithm == "M3":
            logger.warning("The M3 model supports only amp_level O0")
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

    def __call__(self, img_or_path_list: list) -> List:
        """
        Run text classification serially for input images

        Args:
            img_or_path_list: list or str for img path or np.array for RGB image

        Returns:
            list of dict, each contains the follow keys for classification result.
            e.g. [{'angle': 180, 'score': 1.0}, {'angle': 0, 'score': 1.0}]
                - angle: text angle
                - score: prediction confidence
        """

        assert isinstance(
            img_or_path_list, (list, str)
        ), "Input for text classification must be list of images or image paths."
        logger.info(f"num images for cls: {len(img_or_path_list)}")

        if isinstance(img_or_path_list, list):
            cls_res_all_crops = self.run_batch(img_or_path_list)
        else:
            cls_res_all_crops = self.run_single(img_or_path_list)

        return cls_res_all_crops

    def run_batch(self, img_or_path_list: list):
        """
        Run text angle classification serially for input images

        Args:
            img_or_path_list: list of str for img path or np.array for RGB image

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

    def run_single(self, img_or_path: str):
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

        return cls_res

    def save_cls_res(
        self,
        cls_res_all,
        fn="img",
        save_path="./cls_results.txt",
        include_score=True,
    ):
        """
        Generate cls_results files that store the angle classification results.

        Args:
            cls_res_all: list of dict, each contains the follow keys for classification result.
            fn: customize the prefix name for image information, default is "img".
            save_path: file storage path
            include_score: whether to write prediction confidence

        Return:
            lines: the content of angle information written to the document
        """

        lines = []
        for i, cls_res in enumerate(cls_res_all):
            if include_score:
                img_pred = f"{fn}_crop_{i}" + "\t" + cls_res[0] + "\t" + str(cls_res[1]) + "\n"
            else:
                img_pred = f"{fn}_crop_{i}" + "\t" + cls_res[0] + "\n"
            lines.append(img_pred)

        with open(save_path, "a", encoding="utf-8") as f_cls:
            f_cls.writelines(lines)
            f_cls.close()


class TextSystem(object):
    def __init__(self, args):
        self.text_detect = TextDetector(args)
        self.text_recognize = TextRecognizer(args)

        self.cls_algorithm = args.cls_algorithm
        if self.cls_algorithm is not None:
            self.text_classification = TextClassifier(args)
            self.save_cls_result = args.save_cls_result
            self.save_cls_dir = args.crop_res_save_dir

        self.box_type = args.det_box_type
        self.drop_score = args.drop_score
        self.save_crop_res = args.save_crop_res
        self.crop_res_save_dir = args.crop_res_save_dir
        if self.save_crop_res:
            os.makedirs(self.crop_res_save_dir, exist_ok=True)
        self.vis_dir = args.draw_img_save_dir
        os.makedirs(self.vis_dir, exist_ok=True)
        self.vis_font_path = args.vis_font_path

    def __call__(self, img_or_path: Union[str, np.ndarray], do_visualize=True):
        """
        Detect and recognize texts in an image

        Args:
            img_or_path (str or np.ndarray): path to image or image rgb values as a numpy array

        Return:
            boxes (list): detected text boxes, in shape [num_boxes, num_points, 2], where the point coordinate (x, y)
                follows: x - horizontal (image width direction), y - vertical (image height)
            texts (list[tuple]): list of (text, score) where text is the recognized text string for each box,
                and score is the confidence score.
            time_profile (dict): record the time cost for each sub-task.
        """
        assert isinstance(img_or_path, str) or isinstance(
            img_or_path, np.ndarray
        ), "Input must be string of path to the image or numpy array of the image rgb values."
        fn = os.path.basename(img_or_path).rsplit(".", 1)[0] if isinstance(img_or_path, str) else "img"

        time_profile = {}
        start = time()

        # detect text regions on an image
        det_res, data = self.text_detect(img_or_path, do_visualize=False)
        time_profile["det"] = time() - start
        polys = det_res["polys"].copy()
        logger.info(f"Num detected text boxes: {len(polys)}\nDet time: {time_profile['det']}")

        # crop text regions
        crops = []
        for i in range(len(polys)):
            poly = polys[i].astype(np.float32)
            cropped_img = crop_text_region(data["image_ori"], poly, box_type=self.box_type)
            crops.append(cropped_img)

            if self.save_crop_res:
                cv2.imwrite(os.path.join(self.crop_res_save_dir, f"{fn}_crop_{i}.jpg"), cropped_img)
        # show_imgs(crops, is_bgr_img=False)

        if self.cls_algorithm is not None:
            img_or_path = crops
            ct = time()
            cls_res_all = self.text_classification(img_or_path)
            time_profile["cls"] = time() - ct

            cls_count = 0
            for i, cls_res in enumerate(cls_res_all):
                if cls_res[0] != "0":
                    crops[i] = img_rotate(crops[i], -int(cls_res[0]))
                    cls_count = cls_count + 1

            logger.info(
                f"The number of images corrected by rotation is {cls_count}/{len(cls_res_all)}"
                f"\nCLS time: {time_profile['cls']}"
            )

            if self.save_cls_result:
                os.makedirs(self.crop_res_save_dir, exist_ok=True)
                save_fp = os.path.join(self.save_cls_dir, "cls_results.txt")
                self.text_classification.save_cls_res(cls_res_all, fn=fn, save_path=save_fp)

        # recognize cropped images
        rs = time()
        rec_res_all_crops = self.text_recognize(crops, do_visualize=False)
        time_profile["rec"] = time() - rs

        logger.info(
            "Recognized texts: \n"
            + "\n".join([f"{text}\t{score}" for text, score in rec_res_all_crops])
            + f"\nRec time: {time_profile['rec']}"
        )

        # filter out low-score texts and merge detection and recognition results
        boxes, text_scores = [], []
        for i in range(len(polys)):
            box = det_res["polys"][i]
            # box_score = det_res["scores"][i]
            text = rec_res_all_crops[i][0]
            text_score = rec_res_all_crops[i][1]
            if text_score >= self.drop_score:
                boxes.append(box)
                text_scores.append((text, text_score))

        time_profile["all"] = time() - start

        # visualize the overall result
        if do_visualize:
            vst = time()
            vis_fp = os.path.join(self.vis_dir, fn + "_res.png")
            # TODO: improve vis for leaning texts
            visualize(
                data["image_ori"],
                boxes,
                texts=[x[0] for x in text_scores],
                vis_font_path=self.vis_font_path,
                display=False,
                save_path=vis_fp,
                draw_texts_on_blank_page=False,
            )  # NOTE: set as you want
            time_profile["vis"] = time() - vst
        return boxes, text_scores, time_profile


def save_res(boxes_all, text_scores_all, img_paths, save_path="system_results.txt"):
    lines = []
    for i, img_path in enumerate(img_paths):
        # fn = os.path.basename(img_path).split('.')[0]
        boxes = boxes_all[i]
        text_scores = text_scores_all[i]

        res = []  # result for current image
        for j in range(len(boxes)):
            res.append(
                {
                    "transcription": text_scores[j][0],
                    "points": np.array(boxes[j]).astype(np.int32).tolist(),
                }
            )

        img_res_str = os.path.basename(img_path) + "\t" + json.dumps(res, ensure_ascii=False) + "\n"
        lines.append(img_res_str)

    with open(save_path, "w") as f:
        f.writelines(lines)
        f.close()


def main():
    # parse args
    args = parse_args()
    set_logger(name="mindocr")
    save_dir = args.draw_img_save_dir
    img_paths = get_image_paths(args.image_dir)

    # uncomment it to quick test the infer FPS
    # img_paths = img_paths[:10]

    ms.set_context(mode=args.mode)

    # init text system with detector and recognizer
    text_spot = TextSystem(args)

    # warmup
    if args.warmup:
        for i in range(2):
            text_spot(img_paths[0], do_visualize=False)

    # run
    tot_time = {}  # {'det': 0, 'rec': 0, 'all': 0}
    boxes_all, text_scores_all = [], []
    for i, img_path in enumerate(img_paths):
        logger.info(f"\nINFO: Infering [{i+1}/{len(img_paths)}]: {img_path}")
        boxes, text_scores, time_prof = text_spot(img_path, do_visualize=args.visualize_output)
        boxes_all.append(boxes)
        text_scores_all.append(text_scores)

        for k in time_prof:
            if k not in tot_time:
                tot_time[k] = time_prof[k]
            else:
                tot_time[k] += time_prof[k]

    fps = len(img_paths) / tot_time["all"]
    logger.info(f"Total time:{tot_time['all']}")
    logger.info(f"Average FPS: {fps}")
    avg_time = {k: tot_time[k] / len(img_paths) for k in tot_time}
    logger.info(f"Averge time cost: {avg_time}")

    # save result
    save_res(boxes_all, text_scores_all, img_paths, save_path=os.path.join(save_dir, "system_results.txt"))
    logger.info(f"Done! Results saved in {os.path.join(save_dir, 'system_results.txt')}")


if __name__ == "__main__":
    main()
