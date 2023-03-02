import argparse
import os

from deploy.infer_pipeline.framework.module_data_type import InferModelComb


def get_args():
    parser = argparse.ArgumentParser(description='Arguments for inference.')
    # TODO: add default values like dir and path
    parser.add_argument('--device', type=str, default='Ascend310P3', required=False,
                        help='Device type.')  # TODO: add choices?
    parser.add_argument('--device_id', type=int, nargs='+', default=0, required=False, help='Device id.')
    parser.add_argument('--parallel_num', type=str, default=1, required=False, help='Number of parallel inference.')
    parser.add_argument('--precision_mode', type=str, choices=['fp16', 'fp32'], required=False, help='Precision mode.')

    parser.add_argument('--input_images_dir', type=str, required=True,
                        help='Input images dir for inference, can be dir containing multiple images or path of single '
                             'image.')
    parser.add_argument('--det_algorithm', type=str, default='DBNet', required=False, help='Detection algorithm name.')
    parser.add_argument('--cls_algorithm', type=str, default='', required=False, help='Classification algorithm name.')
    parser.add_argument('--rec_algorithm', type=str, default='CRNN', required=False, help='Recognition algorithm name.')
    parser.add_argument('--det_model_path', type=str, required=False, help='Detection model file path.')
    parser.add_argument('--cls_model_path', type=str, default='', required=False,
                        help='Classification model file path.')
    parser.add_argument('--rec_model_path', type=str, required=False, help='Recognition model file path.')
    parser.add_argument('--rec_char_dict_path', type=str, default='', required=False,
                        help='Character dict file path fo r recognition models.')

    parser.add_argument('--det_res_save_dir', type=str, default='', required=False,
                        help='Saving dir for detection results.')
    parser.add_argument('--rec_res_save_dir', type=str, default='', required=False,
                        help='Saving dir for recognition results.')
    parser.add_argument('--pipeline_res_save_dir', type=str, default='', required=False,
                        help='Saving dir for pipeline inference results.')
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


def get_task_type(args):
    is_cls_model_path_exist = os.path.exists(args.cls_model_path)
    is_det_model_path_exist = os.path.exists(args.det_model_path)
    is_rec_model_path_exist = os.path.exists(args.rec_model_path)

    if is_cls_model_path_exist:
        if is_det_model_path_exist:
            if is_rec_model_path_exist:
                return InferModelComb.CLS_DET_REC
            else:
                return InferModelComb.CLS_DET
        else:
            if is_rec_model_path_exist:
                return InferModelComb.CLS_REC
            else:
                raise ValueError(f'Error! Classifier model can not run independently!')
    else:
        if is_det_model_path_exist:
            if is_rec_model_path_exist:
                return InferModelComb.DET_REC
            else:
                return InferModelComb.DET
        else:
            if is_rec_model_path_exist:
                return InferModelComb.REC
            else:
                raise ValueError(f'Error! All model path is not exist!')
