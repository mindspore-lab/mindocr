import argparse
import os
import itertools

from deploy.mindx.framework.module_data_type import InferModelComb
from deploy.mindx.processors import SUPPORT_DET_MODEL, SUPPORT_REC_MODEL


def get_args():
    parser = argparse.ArgumentParser(description='Arguments for inference.')
    parser.add_argument('--input_images_dir', type=str, required=True,
                        help='Input images dir for inference, can be dir containing multiple images or path of single '
                             'image.')

    parser.add_argument('--device', type=str, default='Ascend310P3', required=False,
                        choices=['Ascend310', 'Ascend310P3'], help='Device type.')
    parser.add_argument('--device_id', type=int, default=0, required=False, help='Device id.')
    parser.add_argument('--parallel_num', type=int, default=1, required=False, help='Number of parallel inference.')
    parser.add_argument('--precision_mode', type=str, choices=['fp16', 'fp32'], required=False, help='Precision mode.')

    parser.add_argument('--det_algorithm', type=str, default='DBNet', required=False, help='Detection algorithm name.')
    parser.add_argument('--rec_algorithm', type=str, default='CRNN', required=False, help='Recognition algorithm name.')

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
                        help='Saving dir for visualization of pipeline inference results.')
    parser.add_argument('--vis_font_path', type=str, default='', required=False,
                        help='Font file path for recognition model.')
    parser.add_argument('--save_pipeline_crop_res', type=bool, default=False, required=False,
                        help='Whether save the images cropped during pipeline.')
    parser.add_argument('--pipeline_crop_save_dir', type=str, required=False,
                        help='Saving dir for images cropped during pipeline.')

    parser.add_argument('--show_log', type=bool, default=False, required=False,
                        help='Whether show log when inferring.')
    parser.add_argument('--save_log_dir', type=str, required=False, help='Log saving dir.')

    args = parser.parse_args()
    update_env_os(args)
    args = update_task_args(args)
    check_args(args)
    return args


def update_env_os(args):
    if not args.show_log:
        os.environ['MINDOCR_LOG_LEVEL'] = '2'  # WARNING
    else:
        os.environ['MINDOCR_LOG_LEVEL'] = '1'  # INFO

    if args.save_log_dir:
        os.environ['MINDOCR_LOG_SAVE_PATH'] = args.save_log_dir


def update_task_args(args):
    det = os.path.exists(args.det_model_path) if isinstance(args.det_model_path, str) else False
    cls = os.path.exists(args.cls_model_path) if isinstance(args.cls_model_path, str) else False
    rec = os.path.exists(args.rec_model_path) if isinstance(args.rec_model_path, str) else False

    task_map = {
        (True, False, False): InferModelComb.DET,
        (False, False, True): InferModelComb.REC,
        (True, False, True): InferModelComb.DET_REC,
        (True, True, True): InferModelComb.DET_CLS_REC
    }

    task_order = (det, cls, rec)
    if task_order in task_map:
        taks_type = task_map[task_order]
        setattr(args, 'task_type', taks_type)
    else:
        if not (det or cls or rec):
            raise ValueError(f"det_model_path, cls_model_path, rec_model_path cannot be empty at the same time.")
        elif det:
            raise ValueError(f"rec_model_path can't be empty when det_model_path and cls_model_path are not empty.")
        else:
            raise ValueError(f"cls_model_path{args.cls_model_path} model does not support inference independently.")

    setattr(args, 'save_vis_det_save_dir', True if args.vis_det_save_dir else False)
    setattr(args, 'save_vis_pipeline_save_dir', True if args.vis_pipeline_save_dir else False)

    return args


def check_args(args):
    if not args.input_images_dir or \
            (not os.path.isfile(args.input_images_dir) and not os.path.isdir(args.input_images_dir)) or \
            (os.path.isdir(args.input_images_dir) and len(os.listdir(args.input_images_dir)) == 0):
        raise ValueError(f"input_images_dir must be dir containing multiple images or path of single image.")

    if args.det_model_path and not os.path.isfile(args.det_model_path):
        raise ValueError(f"det_model_path must be a model file path for detection.")

    if args.cls_model_path and not os.path.isfile(args.cls_model_path):
        raise ValueError(f"cls_model_path must be a model file path for classification.")

    if args.rec_model_path and (os.path.isdir(args.rec_model_path) and len(os.listdir(args.rec_model_path)) == 0):
        raise ValueError(f"rec_model_path must a model file or dir containing model file for recognition model.")

    if args.rec_model_path and (not args.rec_char_dict_path or not os.path.isfile(args.rec_char_dict_path)):
        raise ValueError(
            f"rec_char_dict_path must be a dict file for recognition model, but got '{args.rec_char_dict_path}'.")

    if args.parallel_num < 1 or args.parallel_num > 4:
        raise ValueError(f"parallel_num must between [1,4], current: {args.parallel_num}.")

    if args.save_pipeline_crop_res and not args.pipeline_crop_save_dir:
        raise ValueError(f"pipeline_crop_save_dir can’t be empty when save_pipeline_crop_res=True.")

    if args.save_pipeline_crop_res and args.task_type not in (InferModelComb.DET_REC, InferModelComb.DET_CLS_REC):
        raise ValueError(f"det_model_path and rec_model_path can’t be empty when save_pipeline_crop_res=True.")

    if args.vis_pipeline_save_dir and args.task_type not in (InferModelComb.DET_REC, InferModelComb.DET_CLS_REC):
        raise ValueError(f"det_model_path and rec_model_path can’t be empty when set vis_pipeline_save_dir.")

    if args.vis_det_save_dir and args.task_type != InferModelComb.DET:
        raise ValueError(
            f"det_model_path can't be empty and cls_model_path/rec_model_path must be empty when set vis_det_save_dir "
            f"for single detection task.")

    if not args.res_save_dir:
        raise ValueError(f"res_save_dir can’t be empty.")

    if args.det_algorithm not in SUPPORT_DET_MODEL:
        raise ValueError(f"det_algorithm only support {SUPPORT_DET_MODEL}, but got {args.det_algorithm}.")

    if args.rec_algorithm not in SUPPORT_REC_MODEL:
        raise ValueError(f"rec_algorithm only support {SUPPORT_REC_MODEL}, but got {args.rec_algorithm}.")

    check_dir_not_same = {
        "input_images_dir": args.input_images_dir,
        "pipeline_crop_save_dir": args.pipeline_crop_save_dir,
        "vis_pipeline_save_dir": args.vis_pipeline_save_dir,
        "vis_det_save_dir": args.vis_det_save_dir
    }
    for (name1, dir1), (name2, dir2) in itertools.combinations(check_dir_not_same.items(), 2):
        if (dir1 and dir2) and os.path.realpath(os.path.normcase(dir1)) == os.path.realpath(os.path.normcase(dir2)):
            raise ValueError(f"{name1} and {name2} can't be same path.")
