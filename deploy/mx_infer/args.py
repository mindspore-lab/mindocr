import argparse
import itertools
import os

from deploy.mx_infer.framework.module_data_type import InferModelComb
from deploy.mx_infer.processors import SUPPORT_DET_MODEL, SUPPORT_REC_MODEL
from deploy.mx_infer.utils import log


def str2bool(v):
    return v.lower() in ("true", "t", "1")


def get_args():
    """
    command line parameters for inference
    """
    parser = argparse.ArgumentParser(description='Arguments for inference.')
    parser.add_argument('--input_images_dir', type=str, required=True,
                        help='Input images dir for inference, can be dir containing multiple images or path of single '
                             'image.')

    parser.add_argument('--device', type=str, default='Ascend', required=False,
                        choices=['Ascend'], help='Device type.')
    parser.add_argument('--device_id', type=int, default=0, required=False, help='Device id.')
    parser.add_argument('--parallel_num', type=int, default=1, required=False, help='Number of parallel in each stage of pipeline parallelism.')
    parser.add_argument('--precision_mode', type=str, default="fp32", choices=['fp16', 'fp32'], required=False,
                        help='Precision mode.')
    parser.add_argument('--det_algorithm', type=str, default='DBNet', choices=SUPPORT_DET_MODEL, required=False,
                        help='Detection algorithm name.')
    parser.add_argument('--rec_algorithm', type=str, default='CRNN', choices=SUPPORT_REC_MODEL, required=False,
                        help='Recognition algorithm name.')

    parser.add_argument('--det_model_path', type=str, required=False, help='Detection model file path.')
    parser.add_argument('--cls_model_path', type=str, required=False, help='Classification model file path.')
    parser.add_argument('--rec_model_path', type=str, required=False, help='Recognition model file path or directory.')
    parser.add_argument('--rec_char_dict_path', type=str, required=False,
                        help='Character dict file path for recognition models.')

    parser.add_argument('--res_save_dir', type=str, default='inference_results', required=False,
                        help='Saving dir for inference results.')

    parser.add_argument('--vis_det_save_dir', type=str, required=False,
                        help='Saving dir for visualization of detection results.')
    parser.add_argument('--vis_pipeline_save_dir', type=str, required=False,
                        help='Saving dir for visualization of  det+cls(optional)+rec pipeline inference results.')
    parser.add_argument('--vis_font_path', type=str, required=False,
                        help='Font file path for recognition model.')
    parser.add_argument('--pipeline_crop_save_dir', type=str, required=False,
                        help='Saving dir for images cropped during pipeline.')
    parser.add_argument('--show_log', type=str2bool, default=False, required=False,
                        help='Whether show log when inferring.')
    parser.add_argument('--save_log_dir', type=str, required=False, help='Log saving dir.')

    args = parser.parse_args()
    setup_logger(args)
    args = update_task_args(args)
    check_args(args)
    return args


def setup_logger(args):
    """
    initialize log system
    """
    log.init_logger(args.show_log, args.save_log_dir)
    log.save_args(args)


def update_task_args(args):
    """
    add internal parameters according to different task type
    """
    if args.det_model_path and not os.path.exists(args.det_model_path):
        raise ValueError(f"The det_model_path of '{args.det_model_path}' does not exist.")
    if args.cls_model_path and not os.path.exists(args.cls_model_path):
        raise ValueError(f"The cls_model_path of '{args.cls_model_path}' does not exist.")
    if args.rec_model_path and not os.path.exists(args.rec_model_path):
        raise ValueError(f"The rec_model_path of '{args.rec_model_path}' does not exist.")

    det = bool(args.det_model_path)
    cls = bool(args.cls_model_path)
    rec = bool(args.rec_model_path)

    task_map = {
        (True, False, False): InferModelComb.DET,
        (False, False, True): InferModelComb.REC,
        (True, False, True): InferModelComb.DET_REC,
        (True, True, True): InferModelComb.DET_CLS_REC
    }

    task_order = (det, cls, rec)
    if task_order in task_map:
        task_type = task_map[task_order]
        setattr(args, 'task_type', task_type)
        setattr(args, 'save_vis_det_save_dir', bool(args.vis_det_save_dir))
        setattr(args, 'save_vis_pipeline_save_dir', bool(args.vis_pipeline_save_dir))
        setattr(args, 'save_pipeline_crop_res', bool(args.pipeline_crop_save_dir))
    else:
        unsupported_task_map = {
            (False, False, False): "empty",
            (True, True, False): "det+cls",
            (False, True, False): "cls",
            (False, True, True): "cls+rec"
        }

        raise ValueError(
            f"Only support det, rec, det+rec and det+cls+rec, but got {unsupported_task_map[task_order]}. "
            f"Please check model_path!")

    return args


def check_args(args):
    """
    check parameters
    """
    if not args.input_images_dir or not os.path.exists(args.input_images_dir):
        raise ValueError(f"input_images_dir must be dir containing multiple images or path of single image.")

    if args.det_model_path and not os.path.isfile(args.det_model_path):
        raise ValueError(f"det_model_path must be a model file path for detection.")

    if args.cls_model_path and not os.path.isfile(args.cls_model_path):
        raise ValueError(f"cls_model_path must be a model file path for classification.")

    if args.rec_model_path and (not os.path.exists(args.rec_model_path) or (
            os.path.isdir(args.rec_model_path) and not os.listdir(args.rec_model_path))):
        raise ValueError(f"rec_model_path must be a model file or dir containing model file for recognition model.")

    if args.rec_model_path and (not args.rec_char_dict_path or not os.path.isfile(args.rec_char_dict_path)):
        raise ValueError(
            f"rec_char_dict_path must be a dict file for recognition model, but got '{args.rec_char_dict_path}'.")

    if args.parallel_num < 1 or args.parallel_num > 4:
        raise ValueError(f"parallel_num must between [1,4], current: {args.parallel_num}.")

    if args.pipeline_crop_save_dir and args.task_type not in (InferModelComb.DET_REC, InferModelComb.DET_CLS_REC):
        raise ValueError(f"det_model_path and rec_model_path can't be empty when set pipeline_crop_save_dir.")

    if args.vis_pipeline_save_dir and args.task_type not in (InferModelComb.DET_REC, InferModelComb.DET_CLS_REC):
        raise ValueError(f"det_model_path and rec_model_path can't be empty when set vis_pipeline_save_dir.")

    if args.vis_det_save_dir and args.task_type != InferModelComb.DET:
        raise ValueError(
            f"det_model_path can't be empty and cls_model_path/rec_model_path must be empty when set vis_det_save_dir "
            f"for single detection task.")

    if not args.res_save_dir:
        raise ValueError(f"res_save_dir can't be empty.")

    if args.precision_mode != "fp32":
        raise ValueError(f"precision_mode only support fp32 currently.")

    check_dir_not_same = {
        "input_images_dir": args.input_images_dir,
        "pipeline_crop_save_dir": args.pipeline_crop_save_dir,
        "vis_pipeline_save_dir": args.vis_pipeline_save_dir,
        "vis_det_save_dir": args.vis_det_save_dir
    }
    for (name1, dir1), (name2, dir2) in itertools.combinations(check_dir_not_same.items(), 2):
        if (dir1 and dir2) and os.path.realpath(os.path.normcase(dir1)) == os.path.realpath(os.path.normcase(dir2)):
            raise ValueError(f"{name1} and {name2} can't be same path.")
