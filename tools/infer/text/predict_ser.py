"""
Infer semantic entities recognition information from images or images with OCR annotations.

Example:
    $ python tools/infer/text/predict_ser.py  --image_dir {path_to_img} --kie_algorithm VI_LAYOUTXLM
"""
import json
import logging
import os
import sys
from time import time

import cv2
import numpy as np
from config import parse_args
from postprocess import Postprocessor
from predict_system import TextSystem
from preprocess import Preprocessor
from utils import get_ckpt_file, get_image_paths, get_ocr_result_paths

from mindspore import Tensor, set_context

from mindocr import build_model  # noqa
from mindocr.utils.logger import set_logger  # noqa
from mindocr.utils.visualize import draw_ser_results

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))


# map algorithm name to model name (which can be checked by `mindocr.list_models()`)
# NOTE: Modify it to add new model for inference.
algo_to_model_name = {
    "VI_LAYOUTXLM": "vi_layoutxlm_ser",
    "LAYOUTXLM": "layoutxlm_ser",
}
logger = logging.getLogger("mindocr")


class SemanticEntityRecognition:
    """
    Infer model for semantic entity recognition
    """

    def __init__(self, args):
        if args.ocr_result_dir is None:
            self.ocr_system = TextSystem(args)
        self.img_dir = os.path.dirname(args.image_dir)

        # build model for algorithm with pretrained weights or local checkpoint
        ckpt_dir = args.ser_model_dir
        if ckpt_dir is None:
            pretrained = False
            ckpt_load_path = None
            pretrained_backbone = False
        else:
            ckpt_load_path = get_ckpt_file(ckpt_dir)
            pretrained = False
            pretrained_backbone = False
        if args.ser_algorithm not in algo_to_model_name:
            raise ValueError(
                f"Invalid ser algorithm {args.ser_algorithm}. "
                f"Supported ser algorithms are {list(algo_to_model_name.keys())}"
            )
        model_name = algo_to_model_name[args.ser_algorithm]
        self.model = build_model(
            model_name, pretrained=pretrained, pretrained_backbone=pretrained_backbone, ckpt_load_path=ckpt_load_path
        )
        self.model.set_train(False)

        self.preprocess = Preprocessor(
            task="ser",
        )

        self.postprocess = Postprocessor(task="ser")

        self.batch_mode = args.kie_batch_mode
        self.batch_num = args.kie_batch_num

    def format_ocr_result(self, boxes_all, text_scores_all, img_paths):
        """
        Format OCR results for a list of images.

        Args:
            boxes_all (list):
            List of lists, where each inner list contains bounding box coordinates for text in an image.
            text_scores_all (list):
            List of lists, where each inner list contains confidence scores for each detected text in an image.
            img_paths (list): List of file paths to input images.

        Returns:
            list: A list containing dictionaries, where each dictionary represents OCR information for a specific image.
        """
        ocr_info_list = []
        for i, img_path in enumerate(img_paths):
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

            ocr_info = {"img_path": img_path, "label": json.dumps(res)}
            ocr_info_list.append(ocr_info)
        return ocr_info_list

    def get_from_predict(self, img_path_list):
        """
        Perform OCR on a list of input images.

        Args:
            img_path_list (list): A list of file paths to input images that need OCR processing.

        Returns:
            tuple: A tuple containing two elements:
                - res (list): The OCR system's results for each input image. Each element in the list is a dictionary
                            containing OCR information, following the format: {'img_path': img_path, 'label': ocr_info}.
                - tot_time (float): The total time taken by the OCR system to process all input images in seconds.

        Example:
            >>> img_paths = ['/path/to/image1.jpg', '/path/to/image2.jpg']
            >>> results, processing_time = ocr_process(img_paths)
        """

        tot_time = {}  # {'det': 0, 'rec': 0, 'all': 0}
        boxes_all, text_scores_all = [], []
        for i, img_path in enumerate(img_path_list):
            logger.info(f"\nINFO: Infering [{i+1}/{len(img_path_list)}]: {img_path}")
            boxes, text_scores, time_prof = self.ocr_system(img_path, do_visualize=args.visualize_output)
            boxes_all.append(boxes)
            text_scores_all.append(text_scores)

            for k in time_prof:
                if k not in tot_time:
                    tot_time[k] = time_prof[k]
                else:
                    tot_time[k] += time_prof[k]
        res = self.format_ocr_result(boxes_all, text_scores_all, img_path_list)
        return res, tot_time

    def _parse_annotation(self, data_line: str):
        data_line_tmp = data_line.strip()
        if "\t" in data_line_tmp:
            img_name, annot_str = data_line.strip().split("\t")
        elif " " in data_line_tmp:
            img_name, annot_str = data_line.strip().split(" ")
        else:
            raise ValueError(
                "Incorrect label file format: the file name and the label should be separated by " "a space or tab"
            )

        return img_name, annot_str

    def get_from_file(self, label_file_list):
        """
        Load data list from label_file_list which contains information image path and annotation.
        Args:
            label_file_list: annotation file path(s)
        Returns:
            data_list (List[dict]): A list of annotation dict, which contains keys: img_path, annot...
        """

        # parse image file path and annotation and load
        data_list = []
        for label_file in label_file_list:
            with open(label_file, "r", encoding="utf-8") as fp:
                lines = fp.readlines()
            for line in lines:
                img_name, annot_str = self._parse_annotation(line)

                img_path = os.path.join(self.img_dir, img_name)
                if os.path.exists(img_path) is False:
                    raise ValueError(f"{img_name} dose not exist in {self.img_dir}.")

                data = {"img_path": img_path, "label": annot_str}
                data_list.append(data)

        return data_list

    def run_batchwise(self, ocr_info_list):
        """
        Run text recognition serially for input images

        Args:
            ocr_info_list: list of str for img path or np.array for RGB image

        Return:
            ser_res: list of dictionaries, each containing keys 'img_path' and 'ser_output' for recognition results.
        """
        ser_res = []
        num_imgs = len(ocr_info_list)

        for idx in range(0, num_imgs, self.batch_num):  # batch begin index i
            batch_begin = idx
            batch_end = min(idx + self.batch_num, num_imgs)
            input_ids_batch = []
            bbox_batch = []
            attention_mask_batch = []
            token_type_ids_batch = []
            segment_offset_ids_batch = []
            ocr_infos_batch = []
            for j in range(batch_begin, batch_end):  # image index j
                data = self.preprocess(ocr_info_list[j])
                input_ids_batch.append(data["input_ids"])
                bbox_batch.append(data["bbox"])
                attention_mask_batch.append(data["attention_mask"])
                token_type_ids_batch.append(data["token_type_ids"])
                segment_offset_ids_batch.append(data["segment_offset_id"])
                ocr_infos_batch.append(data["ocr_info"])

            input_ids_batch = (
                np.stack(input_ids_batch) if len(input_ids_batch) > 1 else np.expand_dims(input_ids_batch[0], axis=0)
            )
            bbox_batch = np.stack(bbox_batch) if len(bbox_batch) > 1 else np.expand_dims(bbox_batch[0], axis=0)
            attention_mask_batch = (
                np.stack(attention_mask_batch)
                if len(attention_mask_batch) > 1
                else np.expand_dims(attention_mask_batch[0], axis=0)
            )
            token_type_ids_batch = (
                np.stack(token_type_ids_batch)
                if len(token_type_ids_batch) > 1
                else np.expand_dims(token_type_ids_batch[0], axis=0)
            )

            # infer
            input_x = [
                Tensor(input_ids_batch),
                Tensor(bbox_batch),
                Tensor(attention_mask_batch),
                Tensor(token_type_ids_batch),
            ]
            logits = self.model(input_x)
            # postprocess
            batch_res = self.postprocess(logits, segment_offset_ids=segment_offset_ids_batch, ocr_infos=ocr_infos_batch)
            for index, res in enumerate(batch_res):
                res_dict = {"img_path": ocr_info_list[batch_begin + index]["img_path"], "ser_output": res}
                ser_res.append(res_dict)
        return ser_res

    def run_single(self, ocr_info_list):
        """
        Text recognition inference on a single image
        Args:
            ocr_info_list: list of str for img path or np.array for RGB image

        Return:
            dict with keys:
                - texts (str): preditive text string
                - confs (int): confidence of the prediction
        """
        ser_res = []
        # preprocess
        for _, img in enumerate(ocr_info_list):
            data = self.preprocess(img)
            input_ids = data["input_ids"]
            bbox = data["bbox"]
            attention_mask = data["attention_mask"]
            token_type_ids = data["token_type_ids"]
            segment_offset_id = data["segment_offset_id"]
            ocr_info = data["ocr_info"]

            input_ids = np.expand_dims(input_ids, axis=0)
            bbox = np.expand_dims(bbox, axis=0)
            attention_mask = np.expand_dims(attention_mask, axis=0)
            token_type_ids = np.expand_dims(token_type_ids, axis=0)

            input_x = (Tensor(input_ids), Tensor(bbox), Tensor(attention_mask), Tensor(token_type_ids))

            logits = self.model(input_x)

            res = self.postprocess(logits, segment_offset_ids=[segment_offset_id], ocr_infos=[ocr_info])
            res_dict = {"img_path": img["img_path"], "ser_output": res[0]}
            ser_res.append(res_dict)
        return ser_res

    def __call__(self, img_path, ocr_path=None):
        """
        Run text recognition serially for input images.

        Args:
            img_path: Path to an input image or list of image paths.
            ocr_path: Path to OCR results or None.

        Returns:
            list: A list of dictionaries, each containing keys 'img_path' and 'ser_output' for recognition results.
        """
        logger.info(f"num images for ser: {len(img_path)}")
        img_list = get_image_paths(img_path)
        ocr_info_list = []
        if ocr_path is not None:
            logger.info("Get ocr result from file")
            ocr_file_list = get_ocr_result_paths(ocr_path)
            ocr_info_list += self.get_from_file(ocr_file_list)
        else:
            logger.info("Get ocr result from ocr system")
            ocr_info, time_report = self.get_from_predict(img_list)
            ocr_info_list += ocr_info
        start_time = time()
        if self.batch_mode:
            results_ser = self.run_batchwise(ocr_info_list)
        else:
            results_ser = self.run_single(ocr_info_list)
        ser_time = time() - start_time
        time_report["ser"] = ser_time
        return results_ser, time_report


if __name__ == "__main__":
    args = parse_args()
    set_logger(name="mindocr")
    save_dir = args.draw_img_save_dir

    set_context(mode=args.mode)

    # init recognizer
    smt_ent_rec = SemanticEntityRecognition(args)

    # run for each image
    start = time()
    ser_res_all, time_report = smt_ent_rec(args.image_dir, args.ocr_result_dir)
    print(time_report)
    save_dir, _ = os.path.splitext(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    vis = time()
    for ser_res in ser_res_all:
        img = draw_ser_results(ser_res["img_path"], ser_res["ser_output"])
        name, ext = os.path.splitext(os.path.basename(ser_res["img_path"]))
        new_name = name + "_ser" + ext
        new_path = os.path.join(save_dir, new_name)
        cv2.imwrite(new_path, img)
        logger.info(f"Save image: {new_path}")
    vis_time = time() - vis
    all_time = time() - start
    time_report["vis"] = vis_time
    time_report["all"] = all_time
    logger.info(f"\nTotal time: {time_report['all']}")
    logger.info(f"det time: {time_report['det']}")
    logger.info(f"rec time: {time_report['rec']}")
    logger.info(f"ser time: {time_report['ser']}")
    logger.info(f"vis time: {time_report['vis']}")
    logger.info(f"FPS: {len(get_image_paths(args.image_dir))/time_report['all']}")
