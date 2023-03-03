import argparse
import os

from deploy.infer_pipeline.framework.module_data_type import InferModelComb


def get_args():
    parser = argparse.ArgumentParser(description='Arguments for inference.')
    parser.add_argument('--input_images_dir', type=str, required=True,
                        help='Input images dir for inference, can be dir containing multiple images or path of single '
                             'image.')

    parser.add_argument('--device', type=str, default='Ascend310P3', required=False,
                        help='Device type.')  # TODO: add choices?
    parser.add_argument('--device_id', type=int, default=0, required=False, help='Device id.')
    parser.add_argument('--parallel_num', type=int, default=1, required=False, help='Number of parallel inference.')
    parser.add_argument('--precision_mode', type=str, choices=['fp16', 'fp32'], required=False, help='Precision mode.')

    parser.add_argument('--det_algorithm', type=str, default='DBNet', required=False, help='Detection algorithm name.')
    parser.add_argument('--rec_algorithm', type=str, default='CRNN', required=False, help='Recognition algorithm name.')

    parser.add_argument('--det_model_path', type=str, required=False, help='Detection model file path.')
    parser.add_argument('--cls_model_path', type=str, default='', required=False,
                        help='Classification model file path.')
    parser.add_argument('--rec_model_path', type=str, required=False, help='Recognition model file path.')
    parser.add_argument('--rec_char_dict_path', type=str, default='', required=False,
                        help='Character dict file path fo r recognition models.')

    parser.add_argument('--res_save_dir', type=str, default='', required=False,
                        help='Saving dir for inference results.')
    parser.add_argument('--vis_det_save_dir', type=str, default='', required=False,
                        help='Saving dir for visualization of detection results.')
    parser.add_argument('--vis_pipeline_save_dir', type=str, default='', required=False,
                        help='Saving dir for visualization of pipeline inference results.')
    parser.add_argument('--vis_font_path', type=str, default='', required=False,
                        help='Font file path for recognition model.')
    parser.add_argument('--save_pipeline_crop_res', type=bool, default=False, required=False,
                        help='Whether save the images cropped during pipeline.')
    parser.add_argument('--pipeline_crop_save_dir', type=str, default='', required=False,
                        help='Saving dir for images cropped during pipeline.')

    parser.add_argument('--show_log', type=bool, default=True, required=False, help='Whether show log when inferring.')
    parser.add_argument('--save_log_dir', type=str, default='', required=False, help='Log saving dir.')
    return parser.parse_args()


def update_task_type(args):
    det = os.path.exists(args.det_model_path)
    cls = os.path.exists(args.cls_model_path)
    rec = os.path.exists(args.rec_model_path)

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
        if det or cls or rec:
            raise ValueError(f"det_model_path, cls_model_path, rec_model_path cannot be empty at the same time.")
        else:
            raise ValueError(f"cls_model_path{args.cls_model_path} model does not support inference independently.")

    return args

def check_args(args):
    if args.parallel_num < 1 or args.parallel_num > 4:
        raise ValueError(f'parallel num must between [1,4], current: {args.parallel_num}')

