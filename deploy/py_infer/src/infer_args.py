import argparse
import itertools
import os

from .infer import TaskType
from .utils import get_config_by_name_for_model, log, save_path_init


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

    parser.add_argument("--rec_model_path", type=str, required=False, help="Recognition model file path or directory.")
    parser.add_argument(
        "--rec_model_name_or_config", type=str, required=False, help="Recognition model name or config file path."
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
    det = bool(args.det_model_path)
    cls = bool(args.cls_model_path)
    rec = bool(args.rec_model_path)

    task_map = {
        (True, False, False): TaskType.DET,
        (False, True, False): TaskType.CLS,
        (False, False, True): TaskType.REC,
        (True, False, True): TaskType.DET_REC,
        (True, True, True): TaskType.DET_CLS_REC,
    }

    task_order = (det, cls, rec)
    if task_order in task_map:
        setattr(args, "task_type", task_map[task_order])
        setattr(args, "save_vis_det_save_dir", bool(args.vis_det_save_dir))
        setattr(args, "save_vis_pipeline_save_dir", bool(args.vis_pipeline_save_dir))
        setattr(args, "save_crop_res_dir", bool(args.crop_save_dir))
    else:
        unsupported_task_map = {
            (False, False, False): "empty",
            (True, True, False): "det+cls",
            (False, True, True): "cls+rec",
        }

        raise ValueError(
            f"Only support det, cls, rec, det+rec and det+cls+rec, but got {unsupported_task_map[task_order]}. "
            f"Please check model_path!"
        )

    if args.det_model_name_or_config:
        setattr(args, "det_config_path", get_config_by_name_for_model(args.det_model_name_or_config))
    if args.cls_model_name_or_config:
        setattr(args, "cls_config_path", get_config_by_name_for_model(args.cls_model_name_or_config))
    if args.rec_model_name_or_config:
        setattr(args, "rec_config_path", get_config_by_name_for_model(args.rec_model_name_or_config))

    return args


def check_and_update_args(args):
    """
    check parameters
    """
    if not args.input_images_dir or not os.path.exists(args.input_images_dir):
        raise ValueError("input_images_dir must be dir containing multiple images or path of single image.")

    if args.det_model_path and not args.det_model_name_or_config:
        raise ValueError("det_model_name_or_config can't be emtpy when set det_model_path for detection.")

    if args.cls_model_path and not args.cls_model_name_or_config:
        raise ValueError("cls_model_name_or_config can't be emtpy when set cls_model_path for classification.")

    if args.rec_model_path and not args.rec_model_name_or_config:
        raise ValueError("rec_model_name_or_config can't be emtpy when set rec_model_path for recognition.")

    if args.parallel_num < 1 or args.parallel_num > 4:
        raise ValueError(f"parallel_num must between [1,4], current: {args.parallel_num}.")

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
        save_path_init(args.res_save_dir)
    if args.save_crop_res_dir:
        save_path_init(args.crop_save_dir)
    if args.save_vis_pipeline_save_dir:
        save_path_init(args.vis_pipeline_save_dir)
    if args.save_vis_det_save_dir:
        save_path_init(args.vis_det_save_dir)
    if args.save_log_dir:
        save_path_init(args.save_log_dir, exist_ok=True)
