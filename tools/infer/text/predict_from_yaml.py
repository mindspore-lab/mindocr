"""
Inference base on custom yaml

Example:
    $ python tools/infer/text/predict_from_yaml.py  --config configs/det/dbnet/dbpp_r50_icdar15.yaml
    $ python tools/infer/text/predict_from_yaml.py  --config configs/rec/crnn/crnn_resnet34.yaml
"""
import argparse
import logging
import os
import sys
from time import time

import cv2
import numpy as np
import yaml
from addict import Dict
from PIL import Image
from predict_det import save_det_res, validate_det_res
from predict_rec import save_rec_res
from predict_system import save_res
from tqdm import tqdm
from utils import crop_text_region

from mindspore import Tensor, get_context, set_auto_parallel_context, set_context
from mindspore.communication import get_group_size, get_rank, init

from deploy.py_infer.src.infer_args import str2bool  # noqa
from mindocr.data import build_dataset
from mindocr.data.transforms import create_transforms, run_transforms
from mindocr.models import build_model
from mindocr.postprocess import build_postprocess
from mindocr.utils.visualize import draw_boxes, show_imgs
from tools.arg_parser import _merge_options, _parse_options
from tools.infer.text.utils import get_image_paths
from tools.modelarts_adapter.modelarts import modelarts_setup

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))

from mindocr.utils.logger import set_logger  # noqa

logger = logging.getLogger("mindocr")


def draw_det_res(det_res_list: list, img_list: list, output_save_dir: str):
    """
    Draw detection results on images and save the visualized results.

    Args:
        - det_res_list (list): List of dictionaries containing detection results.
        - img_list (list): List of image paths corresponding to the detection results.
        - output_save_dir (str): Directory to save the visualized detection results.

    Returns:
        None
    """
    for det_res, img_path in zip(det_res_list, img_list):
        img = det_res["img_ori"]
        img_name, _ = os.path.splitext(os.path.basename(img_path))
        det_vis = draw_boxes(img, det_res["polys"], is_bgr_img=False)
        show_imgs(
            [det_vis],
            show=False,
            title=img_name + "_det_res",
            save_path=os.path.join(output_save_dir, img_name + "_det_res.png"),
        )


def save_predict_result(task: str, preds_list: list, output_save_dir: str):
    """
    Save and visualize prediction results based on the specified task.

    Args:
        - task (str): Task type, either "det" (detection) or "rec" (recognition).
        - preds_list (list): List of dictionaries containing prediction results.
        - output_save_dir (str): Directory to save the results.

    Returns:
        None
    """
    if task == "det":
        det_res = []
        img_list = []
        for preds in preds_list:
            for i, single_img_polys in enumerate(preds["polys"]):
                img_path = preds["img_path"][i]
                img_list.append(img_path)
                img_shape = Image.open(img_path).size[::-1]
                single_det_res = {"polys": single_img_polys, "scores": preds["scores"][i]}
                single_det_res = validate_det_res(single_det_res, img_shape=img_shape)
                single_det_res["img_ori"] = preds["img_ori"][i]
                det_res.append(single_det_res)
        draw_det_res(det_res, img_list, output_save_dir=output_save_dir)
        save_det_res(det_res, img_list, save_path=os.path.join(output_save_dir, "det_results.txt"))
    elif task == "rec":
        rec_res = []
        img_list = []
        for preds in preds_list:
            img_path = preds["img_path"]
            rec_res.extend(list(zip(preds["texts"], preds["confs"])))
            img_list.extend(img_path)
        save_rec_res(rec_res, img_list, save_path=os.path.join(output_save_dir, "rec_results.txt"))


def predict_single_step(cfg, save_res=True):
    """Run predict for det task or rec task"""
    # 1. Set the environment information.
    set_context(mode=cfg.system.mode)
    output_save_dir = cfg.predict.output_save_dir or "./output"
    os.makedirs(output_save_dir, exist_ok=True)
    if cfg.system.distribute:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        set_auto_parallel_context(
            device_num=device_num,
            parallel_mode="data_parallel",
            gradients_mean=True,
        )
        set_logger(
            name="mindocr", output_dir=cfg.train.ckpt_save_dir or "./", log_fn=f"log_eval_{rank_id}.txt", rank=rank_id
        )
    else:
        device_num = None
        rank_id = None
        set_logger(name="mindocr", output_dir=cfg.train.ckpt_save_dir or "./", log_fn=f"log_eval_{rank_id}.txt", rank=0)
        if "DEVICE_ID" in os.environ:
            logger.info(
                f"Standalone evaluation. Device id: {os.environ.get('DEVICE_ID')}, "
                f"specified by environment variable 'DEVICE_ID'."
            )
        else:
            device_id = cfg.system.get("device_id", 0)
            set_context(device_id=device_id)
            logger.info(
                f"Standalone evaluation. Device id: {device_id}, "
                f"specified by system.device_id in yaml config file or is default value 0."
            )
    # 2. Configuration dataset with pre-processing transform_pipeline
    # Update the configuration for the DecodeImage transform if present
    for transform in cfg.predict.dataset.transform_pipeline:
        if "DecodeImage" in transform:
            transform["DecodeImage"].update({"keep_ori": True})
            break

    # Adjust output columns based on the model type (det or rec)
    if cfg.model.type == "det":
        cfg.predict.dataset.output_columns.extend({"img_path", "image_ori"} - set(cfg.predict.dataset.output_columns))
    elif cfg.model.type == "rec":
        cfg.predict.dataset.output_columns.extend({"img_path"} - set(cfg.predict.dataset.output_columns))

    loader_predict = build_dataset(
        cfg.predict.dataset,
        cfg.predict.loader,
        num_shards=device_num,
        shard_id=rank_id,
        is_train=False,
        refine_batch_size=True,
    )

    # 3.Build model
    network = build_model_from_config(cfg)

    # 4.Build postprocessor for network output
    postprocessor = build_postprocess(cfg.postprocess)

    # 5.Initialize the iterator
    iterator = loader_predict.create_tuple_iterator(num_epochs=1, output_numpy=False, do_copy=False)
    num_batches_predict = loader_predict.get_dataset_size()

    output_columns = cfg.predict.dataset.output_columns or []
    input_indices = cfg.predict.dataset.pop("net_input_column_index", None)
    meta_data_indices = cfg.predict.dataset.pop("meta_data_column_index", None)

    # 6.Start prediction
    logger.info(f"Start {cfg.model.type}")
    preds_list = []
    for i, data in tqdm(enumerate(iterator), total=num_batches_predict):
        if input_indices is not None:
            inputs = [data[x] for x in input_indices]
        else:
            inputs = [data[0]]
        preds = network(*inputs)

        data_info = {"img_shape": inputs[0].shape}
        if meta_data_indices is not None:
            meta_info = [data[x] for x in meta_data_indices]
        else:
            # assume the indices not in input_indices or label_indices are all meta_data_indices
            input_indices = set(input_indices) if input_indices is not None else {0}
            meta_data_indices = sorted(set(range(len(data))) - input_indices)
            meta_info = [data[x] for x in meta_data_indices]

        data_info["meta_info"] = meta_info

        possible_keys_for_postprocess = ["shape_list", "raw_img_shape"]
        for k in possible_keys_for_postprocess:
            if k in output_columns:
                data_info[k] = data[output_columns.index(k)]

        preds = postprocessor(preds, **data_info)
        preds["img_path"] = data[output_columns.index("img_path")].numpy()
        # Add "img_ori" to preds if present, which means task is det
        if "image_ori" in output_columns:
            preds["img_ori"] = data[output_columns.index("image_ori")].numpy()
        if "polys" in preds:
            preds["crops"] = []
            polys_batch = preds["polys"].copy()
            for i, polys in enumerate(polys_batch):
                crops_per_img = []
                for poly in polys:
                    cropped_img = crop_text_region(preds["img_ori"][i], poly, box_type=cfg.postprocess.box_type)
                    crops_per_img.append(cropped_img)
                preds["crops"].append(crops_per_img)
        preds_list.append(preds)

    # 7. Save the prediction results to the specified directory
    if save_res is True:
        save_predict_result(cfg.model.type, preds_list, output_save_dir)
    return preds_list


def build_model_from_config(cfg):
    amp_level = cfg.system.get("amp_level_infer", "O0")
    if get_context("device_target") == "GPU" and amp_level == "O3":
        logger.warning(
            "Model evaluation does not support amp_level O3 on GPU currently. "
            "The program has switched to amp_level O2 automatically."
        )
        amp_level = "O2"
    cfg.model.backbone.pretrained = False
    if cfg.predict.ckpt_load_path is None:
        logger.warning(
            f"No ckpt is available for {cfg.model.task}, "
            "please check your configuration of 'predict.ckpt_load_path' in the yaml file."
        )
    network = build_model(cfg.model, ckpt_load_path=cfg.predict.ckpt_load_path, amp_level=amp_level)
    network.set_train(False)
    return network


def sort_polys(polys):
    return sorted(polys, key=lambda points: (points[0][1], points[0][0]))


def concat_crops(crops: list):
    max_height = max(crop.shape[0] for crop in crops)
    resized_crops = []
    for crop in crops:
        h, w, c = crop.shape
        new_h = max_height
        new_w = int((w / h) * new_h)

        resized_img = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized_crops.append(resized_img)
    crops = np.concatenate(resized_crops, axis=1)
    return crops


class Predict_System:
    def __init__(self, det_cfg, rec_cfg, is_concat=False):
        for transform in det_cfg.predict.dataset.transform_pipeline:
            if "DecodeImage" in transform:
                transform["DecodeImage"].update({"keep_ori": True})
                break
        self.det_transforms = create_transforms(det_cfg.predict.dataset.transform_pipeline)
        self.det_model = build_model_from_config(det_cfg)
        self.det_postprocess = build_postprocess(det_cfg.postprocess)

        self.rec_batch_size = rec_cfg.predict.loader.batch_size
        self.rec_preprocess = create_transforms(rec_cfg.predict.dataset.transform_pipeline)
        self.rec_model = build_model_from_config(rec_cfg)
        self.rec_postprocess = build_postprocess(rec_cfg.postprocess)

        self.is_concat = is_concat

    def predict_rec(self, crops: list):
        """
        Run text recognition serially for input images

                Args:
            img_or_path_list: list of str for img path or np.array for RGB image
            do_visualize: visualize preprocess and final result and save them

                Return:
            rec_res: list of tuple, where each tuple is  (text, score) - text recognition result for each input image
                in order.
                    where text is the predicted text string, score is its confidence score.
                    e.g. [('apple', 0.9), ('bike', 1.0)]
        """
        rec_res = []
        num_crops = len(crops)

        for idx in range(0, num_crops, self.rec_batch_size):  # batch begin index i
            batch_begin = idx
            batch_end = min(idx + self.rec_batch_size, num_crops)
            logger.info(f"Rec img idx range: [{batch_begin}, {batch_end})")
            # TODO: set max_wh_ratio to the maximum wh ratio of images in the batch. and update it for resize,
            #  which may improve recognition accuracy in batch-mode
            # especially for long text image. max_wh_ratio=max(max_wh_ratio, img_w / img_h).
            # The short ones should be scaled with a.r. unchanged and padded to max width in batch.

            # preprocess
            # TODO: run in parallel with multiprocessing
            img_batch = []
            for j in range(batch_begin, batch_end):  # image index j
                data = run_transforms({"image": crops[j]}, self.rec_preprocess[1:])
                img_batch.append(data["image"])

            img_batch = np.stack(img_batch) if len(img_batch) > 1 else np.expand_dims(img_batch[0], axis=0)

            # infer
            net_pred = self.rec_model(Tensor(img_batch))

            # postprocess
            batch_res = self.rec_postprocess(net_pred)
            rec_res.extend(list(zip(batch_res["texts"], batch_res["confs"])))

        return rec_res

    def predict(self, img_path):
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

        time_profile = {}
        start = time()

        # detect text regions on an image
        data = {"img_path": img_path}
        data = run_transforms(data, self.det_transforms)
        input_np = np.expand_dims(data["image"], axis=0)
        logits = self.det_model(Tensor(input_np))
        pred = self.det_postprocess(logits, shape_list=np.expand_dims(data["shape_list"], axis=0))
        polys = pred["polys"][0]
        scores = pred["scores"][0]
        pred = dict(polys=polys, scores=scores)
        det_res = validate_det_res(pred, data["image_ori"].shape[:2], min_poly_points=3, min_area=3)
        det_res["img_ori"] = data["image_ori"]

        time_profile["det"] = time() - start
        polys = det_res["polys"].copy()
        if len(polys) == 0:
            logger.warning(f"No text detected in {img_path}")
            time_profile["rec"] = 0.0
            time_profile["all"] = time_profile["det"]
            return [], [], time_profile
        polys = sort_polys(polys)
        logger.info(f"Num detected text boxes: {len(polys)}\nDet time: {time_profile['det']}")
        if self.is_concat:
            logger.info("After concatenating, 1 croped image will be recognized.")

        # crop text regions
        crops = []
        for i in range(len(polys)):
            poly = polys[i].astype(np.float32)
            cropped_img = crop_text_region(data["image_ori"], poly, box_type=det_cfg.postprocess.box_type)
            crops.append(cropped_img)

            # if self.save_crop_res:
            #     cv2.imwrite(os.path.join(self.crop_res_save_dir, f"{fn}_crop_{i}.jpg"), cropped_img)
        # show_imgs(crops, is_bgr_img=False)

        # recognize cropped images
        rs = time()
        if self.is_concat:
            crops = [concat_crops(crops)]
        rec_res_all_crops = self.predict_rec(crops)
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
            if self.is_concat:
                text = rec_res_all_crops[0][0]
                text_score = rec_res_all_crops[0][1]
            else:
                text = rec_res_all_crops[i][0]
                text_score = rec_res_all_crops[i][1]

            if text_score >= 0.5:
                boxes.append(box)
                text_scores.append((text, text_score))
        time_profile["all"] = time() - start
        return boxes, text_scores, time_profile


def predict_both_step(args, det_cfg, rec_cfg):
    # parse args
    set_logger(name="mindocr")
    pred_sys = Predict_System(det_cfg=det_cfg, rec_cfg=rec_cfg, is_concat=args.is_concat)
    output_save_dir = det_cfg.predict.output_save_dir or "./output"
    img_paths = get_image_paths(args.image_dir)

    set_context(mode=det_cfg.system.mode)

    tot_time = {}  # {'det': 0, 'rec': 0, 'all': 0}
    boxes_all, text_scores_all = [], []
    for i, img_path in enumerate(img_paths):
        logger.info(f"Infering [{i+1}/{len(img_paths)}]: {img_path}")
        boxes, text_scores, time_prof = pred_sys.predict(img_path)
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
    save_res(boxes_all, text_scores_all, img_paths, save_path=os.path.join(output_save_dir, "system_results.txt"))
    logger.info(f"Done! Results saved in {os.path.join(output_save_dir, 'system_results.txt')}")


def create_parser():
    parser = argparse.ArgumentParser(description="Training Config", add_help=False)
    parser.add_argument("--image_dir", type=str, help="image path or image directory")
    parser.add_argument("--task_mode", type=str, default="system", choices=["det", "rec", "system"], help="Task mode")
    parser.add_argument(
        "--det_config",
        type=str,
        default="configs/det/dbnet/db_r50_icdar15.yaml",
        help='YAML config file specifying default arguments for det (default="configs/det/dbnet/db_r50_icdar15.yaml")',
    )
    parser.add_argument(
        "--rec_config",
        type=str,
        default="configs/rec/crnn/crnn_resnet34.yaml",
        help='YAML config file specifying default arguments for rec (default="configs/rec/crnn/crnn_resnet34.yaml")',
    )
    parser.add_argument("--is_concat", type=str2bool, default=False, help="image path or image directory")
    parser.add_argument(
        "-o",
        "--opt",
        nargs="+",
        help="Options to change yaml configuration values, "
        "e.g. `-o system.distribute=False eval.dataset.dataset_root=/my_path/to/ocr_data`",
    )
    # modelarts
    group = parser.add_argument_group("modelarts")
    group.add_argument(
        "--enable_modelarts", type=str2bool, default=False, help="Run on modelarts platform (default=False)"
    )
    group.add_argument(
        "--device_target",
        type=str,
        default="Ascend",
        help="Target device, only used on modelarts platform (default=Ascend)",
    )
    # The url are provided by modelart, usually they are S3 paths
    group.add_argument("--multi_data_url", type=str, default="", help="path to multi dataset")
    group.add_argument("--data_url", type=str, default="", help="path to dataset")
    group.add_argument("--ckpt_url", type=str, default="", help="pre_train_model path in obs")
    group.add_argument("--pretrain_url", type=str, default="", help="pre_train_model paths in obs")
    group.add_argument("--train_url", type=str, default="", help="model folder to save/load")

    return parser


def parse_args_and_config():
    """
    Return:
        args: command line argments
        cfg: train/eval config dict
    """
    parser = create_parser()
    args = parser.parse_args()  # CLI args

    modelarts_setup(args)
    if args.task_mode == "system" and args.image_dir is None:
        raise ValueError("When the task is 'ocr', the 'image_dir' is necessary.")

    with open(args.det_config, "r") as f:
        det_cfg = yaml.safe_load(f)
    with open(args.rec_config, "r") as f:
        rec_cfg = yaml.safe_load(f)

    if args.opt:
        options = _parse_options(args.opt)
        det_cfg = _merge_options(det_cfg, options)
        rec_cfg = _merge_options(rec_cfg, options)
    return args, det_cfg, rec_cfg


if __name__ == "__main__":
    args, det_cfg, rec_cfg = parse_args_and_config()
    if args.task_mode == "det":
        det_cfg = Dict(det_cfg)
        predict_single_step(det_cfg)
    elif args.task_mode == "rec":
        rec_cfg = Dict(rec_cfg)
        predict_single_step(rec_cfg)
    elif args.task_mode == "system":
        rec_cfg = Dict(rec_cfg)
        det_cfg = Dict(det_cfg)
        predict_both_step(args, det_cfg, rec_cfg)
