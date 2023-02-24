import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Arguments for inference.')

    parser.add_argument('--device', type=str, default='Ascend310P3', required=False, help='Device type.') #TODO: add choices?
    parser.add_argument('--device_id', type=int, nargs='+', default=0, required=False, help='Device id.')
    parser.add_argument('--parallel_num', type=str, default=1, required=False, help='Number of parallel inference.')
    parser.add_argument('--precision_mode', type=str, choices=['fp16', 'fp32'], required=False, help='Precision mode.') #TODO

    parser.add_argument('--input_images_dir', type=str, required=True, help='Input images dir for inference, can be dir containing multiple images or path of single image.')
    parser.add_argument('--det_algorithm', type=str, default='DBNet', required=False, help='Detection algorithm name.')
    parser.add_argument('--cls_algorithm', type=str, default='', required=False, help='Classification algorithm name.')
    parser.add_argument('--rec_algorithm', type=str, default='CRNN', required=False, help='Recognition algorithm name.')
    parser.add_argument('--det_model_path', type=str, required=False, help='Detection model file path.')
    parser.add_argument('--cls_model_path', type=str, default='', required=False, help='Classification model file path.')
    parser.add_argument('--rec_model_path', type=str, required=False, help='Recognition model file path.')
    parser.add_argument('--rec_char_dict_path', type=str, default='', required=False, help='Character dict file path fo r recognition models.')   #TODO

    parser.add_argument('--det_res_save_dir', type=str, default='', required=False, help='Saving dir for detection results.') #TODO
    parser.add_argument('--rec_res_save_dir', type=str, default='', required=False, help='Saving dir for recognition results.') #TODO
    parser.add_argument('--pipeline_res_save_dir', type=str, default='', required=False, help='Saving dir for pipeline inference results.') #TODO
    parser.add_argument('--vis_det_save_dir', type=str, default='', required=False, help='Saving dir for visualization of detection results.') #TODO
    parser.add_argument('--vis_pipeline_save_dir', type=str, default='', required=False, help='Saving dir for visualization of pipeline inference results.') #TODO
    parser.add_argument('--vis_font_path', type=str, default='', required=False, help='Font file path for recognition model.') #TODO

    parser.add_argument('--show_log', type=bool, default=True, required=False, help='Whether show log when inferring.')
    parser.add_argument('--save_log_dir', type=str, default='', required=False, help='Log saving dir.') #TODO

    return parser.parse_args()
