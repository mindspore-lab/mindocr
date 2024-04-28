import argparse
import itertools
import os
import sys
import yaml

from addict import Dict

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../")))

from pipeline.tasks import TaskType
from pipeline.utils import get_config_by_name_for_model, log, save_path_init


def str2bool(v):
    return v.lower() in ("true", "t", "1")


def get_args():
    """
    command line parameters for inference
    """
    parser = argparse.ArgumentParser(description="Arguments for inference.")
    parser.add_argument(
        "--input_images_dir",
        type=str,
        required=True,
        help="Image or folder path for inference",
    )

    parser.add_argument(
        "--node_fetch_interval",
        type=float,
        default=0.001,
        required=False,
        help="Interval(seconds) that each node fetch data from queue.",
    )

    parser.add_argument(
        "--result_contain_score",
        type=bool,
        default=False,
        required=False,
        help="If save confidence score to output result.",
    )

    parser.add_argument(
        "--det_algorithm",
        type=str,
        default="DB++",
        choices=["DB", "DB++", "DB_MV3", "DB_PPOCRv3", "PSE"],
        help="detection algorithm.",
    )  # determine the network architecture
    parser.add_argument(
        "--det_model_name_or_config", type=str, required=False, help="Detection model name or config file path."
    )

    parser.add_argument(
        "--cls_algorithm",
        type=str,
        default="MV3",
        choices=["MV3"],
        help="classification algorithm.",
    )  # determine the network architecture
    parser.add_argument(
        "--cls_model_name_or_config", type=str, required=False, help="Classification model name or config file path."
    )

    parser.add_argument(
        "--rec_algorithm",
        type=str,
        default="CRNN",
        choices=["CRNN", "RARE", "CRNN_CH", "RARE_CH", "SVTR", "SVTR_PPOCRv3_CH"],
        help="recognition algorithm",
    )
    parser.add_argument(
        "--rec_model_name_or_config", type=str, required=False, help="Recognition model name or config file path."
    )

    parser.add_argument(
        "--layout_algorithm",
        type=str,
        default="YOLOV8",
        choices=["YOLOV8"],
        help="layout algorithm.",
    )  # determine the network architecture
    parser.add_argument(
        "--layout_model_name_or_config", type=str, required=False, help="Layout model name or config file path."
    )

    parser.add_argument(
        "--character_dict_path", type=str, required=False, help="Character dict file path for recognition models."
    )

    # ZHQ TODO
    parser.add_argument(
        "--layout_model_name_or_config", type=str, required=False, help="Layout model name or config file path."
    )

    parser.add_argument(
        "--res_save_dir",
        type=str,
        default="inference_results",
        required=False,
        help="Saving dir for inference results.",
    )

    parser.add_argument(
        "--input_array_save_dir",
        type=str,
        required=False,
        help="Saving input array.",
    )

    parser.add_argument(
        "--vis_det_save_dir", type=str, required=False, help="Saving dir for visualization of detection results."
    )

    parser.add_argument(
        "--vis_layout_save_dir", type=str, required=False, help="Saving dir for visualization of layout results."
    )

    parser.add_argument(
        "--vis_pipeline_save_dir",
        type=str,
        required=False,
        help="Saving dir for visualization of det+cls(optional)+rec pipeline inference results.",
    )
    parser.add_argument(
        "--crop_save_dir", type=str, required=False, help="Saving dir for images cropped of detection results."
    )
    parser.add_argument(
        "--show_log", type=str2bool, default=False, required=False, help="Whether show log when inferring."
    )
    parser.add_argument("--save_log_dir", type=str, required=False, help="Log saving dir.")
    font_default_path = os.path.join(__dir__, "../docs/fonts/simfang.ttf")
    parser.add_argument(
        "--vis_font_path",
        type=str,
        default=font_default_path,
        required=False,
        help="Font file path for recognition model.")
    parser.add_argument(
        "--visual_pipeline",
        type=bool,
        default=True,
        required=False,
        help="visualize pipeline progress.",
    )
    args = parser.parse_args()
    setup_logger(args)
    args = update_task_info(args)
    # check_and_update_args(args)
    init_save_dir(args)

    return args


def setup_logger(args):
    """
    initialize log system
    """
    log.init_logger(args.show_log, args.save_log_dir)
    log.save_args(args)


def update_task_info(args):
    """
    add internal parameters according to different task type
    """
    det = bool(args.det_model_name_or_config)
    cls = bool(args.cls_model_name_or_config)
    rec = bool(args.rec_model_name_or_config)
    layout = bool(args.layout_model_name_or_config)

    task_map = {
        (True, False, False, False): TaskType.DET,
        (False, True, False, False): TaskType.CLS,
        (False, False, True, False): TaskType.REC,
        (True, False, True, False): TaskType.DET_REC,
        (True, True, True, False): TaskType.DET_CLS_REC,
        (False, False, False, True): TaskType.LAYOUT,
        (True, False, True, True): TaskType.LAYOUT_DET_REC,
        (True, True, True, True): TaskType.LAYOUT_DET_CLS_REC,
    }

    task_order = (det, cls, rec, layout)
    if task_order in task_map:
        setattr(args, "task_type", task_map[task_order])
    else:
        unsupported_task_map = {
            (False, False, False, False): "empty",
            (True, True, False, False): "det+cls",
            (False, True, True, False): "cls+rec",
        }

        raise ValueError(
            f"Only support det, cls, rec, det+rec and det+cls+rec, but got {unsupported_task_map[task_order]}. "
            f"Please check model_path!"
        )

    if args.det_model_name_or_config:
        setattr(args, "det_config_path", get_config_by_name_for_model(args.det_model_name_or_config))
    else:
        setattr(args, "det_config_path", None)
    if args.cls_model_name_or_config:
        setattr(args, "cls_config_path", get_config_by_name_for_model(args.cls_model_name_or_config))
    else:
        setattr(args, "cls_config_path", None)
    if args.rec_model_name_or_config:
        setattr(args, "rec_config_path", get_config_by_name_for_model(args.rec_model_name_or_config))
    else:
        setattr(args, "rec_config_path", None)
    if args.layout_model_name_or_config:
        setattr(args, "layout_config_path", get_config_by_name_for_model(args.layout_model_name_or_config))
    else:
        setattr(args, "layout_config_path", None)

    return args

def check_file(name, file):
    if not os.path.exists(file):
        raise ValueError(f"{name} must be a file, but {file} doesn't exist.")
    if not os.path.isfile(file):
        raise ValueError(f"{name} must be a file, but got a dir of {file}.")

def check_positive(name, value):
    if value < 1:
        raise ValueError(f"{name} must be positive, but got {value}.")


def check_and_update_args(args):
    """
    check parameters
    """
    if not args.input_images_dir or not os.path.exists(args.input_images_dir):
        raise ValueError("input_images_dir must be dir containing multiple images or path of single image.")

    if args.crop_save_dir and args.task_type not in (TaskType.DET_REC, TaskType.DET_CLS_REC):
        raise ValueError("det_model_path and rec_model_path can't be empty when set crop_save_dir.")

    if args.vis_pipeline_save_dir and args.task_type not in (TaskType.DET_REC,
                                                             TaskType.DET_CLS_REC, TaskType.LAYOUT_DET_CLS_REC):
        raise ValueError("det_model_path and rec_model_path can't be empty when set vis_pipeline_save_dir.")

    if args.vis_det_save_dir and args.task_type not in (TaskType.DET, TaskType.LAYOUT):
        raise ValueError(
            "det_model_path can't be empty and cls_model_path/rec_model_path must be empty when set vis_det_save_dir "
            "for single detection task."
        )

    if not args.res_save_dir:
        raise ValueError("res_save_dir can't be empty.")

    need_check_file = {
        "det_config_path": args.det_config_path,
        "cls_config_path": args.cls_config_path,
        "rec_config_path": args.rec_config_path,
    }
    for name, path in need_check_file.items():
        if path:
            check_file(name, path)
            with open(path) as fp:
                yaml_cfg = Dict(yaml.safe_load(fp))
                check_file(name, yaml_cfg.predict.ckpt_load_path)
                check_positive(name, yaml_cfg.predict.loader.batch_size)

    need_check_dir_not_same = {
        "input_images_dir": args.input_images_dir,
        "crop_save_dir": args.crop_save_dir,
        "vis_pipeline_save_dir": args.vis_pipeline_save_dir,
        "vis_det_save_dir": args.vis_det_save_dir,
    }
    for (name1, dir1), (name2, dir2) in itertools.combinations(need_check_dir_not_same.items(), 2):
        if (dir1 and dir2) and os.path.realpath(os.path.normcase(dir1)) == os.path.realpath(os.path.normcase(dir2)):
            raise ValueError(f"{name1} and {name2} can't be same path.")

    return args


def init_save_dir(args):
    if args.res_save_dir:
        save_path_init(args.res_save_dir, exist_ok=True)
    if args.crop_save_dir:
        save_path_init(args.crop_save_dir)
    if args.vis_pipeline_save_dir:
        save_path_init(args.vis_pipeline_save_dir)
    if args.vis_det_save_dir:
        save_path_init(args.vis_det_save_dir)
    if args.save_log_dir:
        save_path_init(args.save_log_dir, exist_ok=True)
