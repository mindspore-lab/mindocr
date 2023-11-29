import argparse
import itertools
import os
import sys

mindocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, mindocr_path)

from deploy.py_infer.src.infer_args import str2bool, setup_logger, update_task_info, init_save_dir  # noqa
from deploy.py_infer.src.infer import TaskType

def get_args():
    """
    command line parameters for inference
    """
    parser = argparse.ArgumentParser(description="Arguments for inference.")
    
    parser.add_argument(
        "--backend",
        type=str.lower,
        default="lite",
        required=False,
        choices=["acl", "lite"],
        help="Inference backend type.",
    )
    parser.add_argument("--device", type=str, default="Ascend", required=False, choices=["Ascend"], help="Device type.")
    parser.add_argument("--device_id", type=int, default=0, required=False, help="Device id.")
    parser.add_argument(
        "--parallel_num",
        type=int,
        default=1,
        required=False,
        help="Number of parallel in each stage of pipeline parallelism.",
    )
    parser.add_argument(
        "--precision_mode", type=str, default=None, choices=["fp16", "fp32"], required=False, help="Precision mode."
    )

    parser.add_argument("--det_model_path", type=str, required=False, help="Detection model file path.")
    parser.add_argument(
        "--det_model_name_or_config", type=str, required=False, help="Detection model name or config file path."
    )

    parser.add_argument("--cls_model_path", type=str, required=False, help="Classification model file path.")
    parser.add_argument(
        "--cls_model_name_or_config", type=str, required=False, help="Classification model name or config file path."
    )
    parser.add_argument(
        "--cls_batch_num", type=int, default=6, required=False, help="Batch size for classification model."
    )

    parser.add_argument("--rec_model_path", type=str, required=False, help="Recognition model file path or directory.")
    parser.add_argument(
        "--rec_model_name_or_config", type=str, required=False, help="Recognition model name or config file path."
    )
    parser.add_argument(
        "--rec_batch_num", type=int, default=6, required=False, help="Batch size for recognition model."
    )
    parser.add_argument(
        "--character_dict_path", type=str, required=False, help="Character dict file path for recognition models."
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
        default="input_array_temp",
        required=False,
        help="Saving input array.",
    )

    parser.add_argument(
        "--vis_det_save_dir", type=str, required=False, help="Saving dir for visualization of detection results."
    )
    parser.add_argument(
        "--vis_pipeline_save_dir",
        type=str,
        required=False,
        help="Saving dir for visualization of det+cls(optional)+rec pipeline inference results.",
    )
    parser.add_argument("--vis_font_path", type=str, required=False, help="Font file path for recognition model.")
    parser.add_argument(
        "--crop_save_dir", type=str, required=False, help="Saving dir for images cropped of detection results."
    )
    parser.add_argument(
        "--show_log", type=str2bool, default=False, required=False, help="Whether show log when inferring."
    )
    parser.add_argument("--save_log_dir", type=str, required=False, help="Log saving dir.")

    args = parser.parse_args()
    setup_logger(args)
    args = update_task_info(args)
    check_and_update_args(args)
    init_save_dir(args)

    return args

def check_and_update_args(args):
    """
    check parameters
    """
    if args.det_model_path and not args.det_model_name_or_config:
        raise ValueError("det_model_name_or_config can't be emtpy when set det_model_path for detection.")

    if args.cls_model_path and not args.cls_model_name_or_config:
        raise ValueError("cls_model_name_or_config can't be emtpy when set cls_model_path for classification.")

    if args.rec_model_path and not args.rec_model_name_or_config:
        raise ValueError("rec_model_name_or_config can't be emtpy when set rec_model_path for recognition.")

    if args.crop_save_dir and args.task_type not in (TaskType.DET_REC, TaskType.DET_CLS_REC):
        raise ValueError("det_model_path and rec_model_path can't be empty when set crop_save_dir.")

    if args.vis_pipeline_save_dir and args.task_type not in (TaskType.DET_REC, TaskType.DET_CLS_REC):
        raise ValueError("det_model_path and rec_model_path can't be empty when set vis_pipeline_save_dir.")

    if args.vis_det_save_dir and args.task_type != TaskType.DET:
        raise ValueError(
            "det_model_path can't be empty and cls_model_path/rec_model_path must be empty when set vis_det_save_dir "
            "for single detection task."
        )

    if not args.res_save_dir:
        raise ValueError("res_save_dir can't be empty.")

    need_check_file = {
        "det_model_path": args.det_model_path,
        "cls_model_path": args.cls_model_path,
        "character_dict_path": args.character_dict_path,
    }
    for name, path in need_check_file.items():
        if path:
            if not os.path.exists(path):
                raise ValueError(f"{name} must be a file, but {path} doesn't exist.")
            if not os.path.isfile(path):
                raise ValueError(f"{name} must be a file, but got a dir of {path}.")

    need_check_file_or_dir = {
        "rec_model_path": args.rec_model_path,
    }
    for name, path in need_check_file_or_dir.items():
        if path:
            if not os.path.exists(path):
                raise ValueError(f"{name} must be a file or dir, but {path} doesn't exist.")
            if os.path.isdir(path) and not os.listdir(path):
                raise ValueError(f"{name} got a dir of {path}, but it is empty.")

    need_check_positive = {
        "parallel_num": args.parallel_num,
        "rec_batch_num": args.rec_batch_num,
        "cls_batch_num": args.cls_batch_num,
    }
    for name, value in need_check_positive.items():
        if value < 1:
            raise ValueError(f"{name} must be positive, but got {value}.")

    need_check_dir_not_same = {
        "crop_save_dir": args.crop_save_dir,
        "vis_pipeline_save_dir": args.vis_pipeline_save_dir,
        "vis_det_save_dir": args.vis_det_save_dir,
        "input_array_save_dir": args.input_array_save_dir,
    }
    for (name1, dir1), (name2, dir2) in itertools.combinations(need_check_dir_not_same.items(), 2):
        if (dir1 and dir2) and os.path.realpath(os.path.normcase(dir1)) == os.path.realpath(os.path.normcase(dir2)):
            raise ValueError(f"{name1} and {name2} can't be same path.")

    return args