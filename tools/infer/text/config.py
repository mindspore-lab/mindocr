'''
Arguments for inference.

Argument names are adopted from ppocr for easy usage transfer.
'''
import argparse
import os
import sys


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "1"):
        return True
    elif v.lower() in ("no", "false", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def create_parser():
    parser_config = argparse.ArgumentParser(description='Inference Config File', add_help=False)
    parser_config.add_argument('-c', '--config', type=str, default='',
            help='YAML config file specifying default arguments (default="")')

    parser = argparse.ArgumentParser(description='Inference Config Args')
    # params for prediction engine
    parser.add_argument("--mode", type=int, default=0, help='0 for graph mode, 1 for pynative mode ') #added
    #parser.add_argument("--use_gpu", type=str2bool, default=True)
    #parser.add_argument("--use_npu", type=str2bool, default=False)
    #parser.add_argument("--ir_optim", type=str2bool, default=True)
    #parser.add_argument("--min_subgraph_size", type=int, default=15)
    #parser.add_argument("--precision", type=str, default="fp32")
    #parser.add_argument("--gpu_mem", type=int, default=500)
    #parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--det_model_config", type=str, help='path to det model yaml config') # added
    parser.add_argument("--rec_model_config", type=str, help='path to rec model yaml config') # added

    # params for text detector
    parser.add_argument("--image_dir", type=str, help='image path or image directory')
    #parser.add_argument("--page_num", type=int, default=0)
    parser.add_argument("--det_algorithm", type=str, default='DB++', choices=['DB', 'DB++', 'DB_MV3', 'PSE'],
            help='detection algorithm.') # determine the network architecture
    parser.add_argument("--det_model_dir", type=str, default=None,
            help='directory containing the detection model checkpoint best.ckpt, or path to a specific checkpoint file.') # determine the network weights
    parser.add_argument("--det_limit_side_len", type=int, default=960,
            help="side length limitation for image resizing"
                        ) # increase if need
    parser.add_argument("--det_limit_type", type=str, default='max', choices=['min', 'max'],
            help='limitation type for image resize. If min, images will be resized by limiting the mininum side length to `limit_side_len` (prior to accuracy). If max, images will be resized by limiting the maximum side length to `limit_side_len` (prior to speed). Default: max')
    parser.add_argument("--det_box_type", type=str, default='quad', choices=['quad', 'poly'],
            help='box type for text region representation')


    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=str2bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="fast")

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='CRNN', choices=['CRNN', 'RARE', 'CRNN_CH', 'RARE_CH', 'SVTR'],
            help='recognition algorithm')
    parser.add_argument("--rec_amp_level", type=str, default='O0', choices=['O0', 'O2', 'O3'],
                        help='Auto Mixed Precision level. This setting only works on GPU and Ascend') #added
    parser.add_argument("--rec_model_dir", type=str,
            help='directory containing the recognition model checkpoint best.ckpt, or path to a specific checkpoint file.') # determine the network weights
    #parser.add_argument("--rec_image_inverse", type=str2bool, default=True)
    parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320",
            help='C, H, W for taget image shape. max_wh_ratio=W/H will be used to control the maximum width after "aspect-ratio-kept" resizing. Set W larger for longer text.')

    parser.add_argument("--rec_batch_mode", type=str2bool, default=True,
            help="Whether to run recogintion inference in batch-mode, which is faster but may degrade the accraucy due to padding or resizing to the same shape.") # added
    parser.add_argument("--rec_batch_num", type=int, default=8)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument("--rec_char_dict_path", type=str, default=None,
            help='path to character dictionary. If None, will pick according to rec_algorithm and red_model_dir.')
    #parser.add_argument("--use_space_char", type=str2bool, default=True) # uncomment it after model trained supporting space recognition.
    parser.add_argument(
        "--vis_font_path", type=str, default="docs/fonts/simfang.ttf")
    parser.add_argument("--drop_score", type=float, default=0.5)
    parser.add_argument("--rec_gt_path", type=str, default=None,
            help='Path to ground truth labels of the recognition result') # added

    #
    parser.add_argument("--draw_img_save_dir", type=str, default="./inference_results",
            help='Dir to save visualization and detection/recogintion/system prediction results')
    parser.add_argument("--save_crop_res", type=str2bool, default=False, help='Whether to save images cropped from text detection results.')
    parser.add_argument("--crop_res_save_dir", type=str, default="./output", help='Dir to save the cropped images for text boxes')
    parser.add_argument("--visualize_output", type=str2bool, default=False, help='Whether to visualize results and save the visualized image.')

    # multi-process
    '''
    parser.add_argument("--use_mp", type=str2bool, default=False)
    parser.add_argument("--total_process_num", type=int, default=1)
    parser.add_argument("--process_id", type=int, default=0)

    parser.add_argument("--benchmark", type=str2bool, default=False)
    parser.add_argument("--save_log_path", type=str, default="./log_output/")

    parser.add_argument("--show_log", type=str2bool, default=True)
    parser.add_argument("--use_onnx", type=str2bool, default=False)

    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=10)
    '''
    parser.add_argument("--warmup", type=str2bool, default=False)

    return parser_config, parser


def _check_cfgs_in_parser(cfgs: dict, parser: argparse.ArgumentParser):
    actions_dest = [action.dest for action in parser._actions]
    defaults_key = parser._defaults.keys()
    for k in cfgs.keys():
        if k not in actions_dest and k not in defaults_key:
            raise KeyError(f"{k} does not exist in ArgumentParser!")


def parse_args(args=None):
    parser_config, parser = create_parser()
    # Do we have a config file to parse?
    args_config, remaining = parser_config.parse_known_args(args)
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
            parser.set_defaults(config=args_config.config)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    return args

