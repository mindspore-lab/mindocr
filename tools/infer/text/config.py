"""
Arguments for inference.

Argument names are adopted from ppocr for easy usage transfer.
"""
import argparse


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
    parser = argparse.ArgumentParser(description="Inference Config Args")
    # params for prediction engine
    parser.add_argument("--mode", type=int, default=0, help="0 for graph mode, 1 for pynative mode ")  # added

    # params for text detector
    parser.add_argument("--image_dir", type=str, required=True, help="image path or image directory")
    # parser.add_argument("--page_num", type=int, default=0)
    parser.add_argument(
        "--det_algorithm",
        type=str,
        default="DB++",
        choices=["DB", "DB++", "DB_MV3", "DB_PPOCRv3", "PSE"],
        help="detection algorithm.",
    )  # determine the network architecture
    parser.add_argument(
        "--det_amp_level",
        type=str,
        default="O0",
        choices=["O0", "O1", "O2", "O3"],
        help="Auto Mixed Precision level. This setting only works on GPU and Ascend",
    )
    parser.add_argument(
        "--det_model_dir",
        type=str,
        default=None,
        help="directory containing the detection model checkpoint best.ckpt, or path to a specific checkpoint file.",
    )  # determine the network weights
    parser.add_argument(
        "--det_limit_side_len", type=int, default=960, help="side length limitation for image resizing"
    )  # increase if need
    parser.add_argument(
        "--det_limit_type",
        type=str,
        default="max",
        choices=["min", "max"],
        help="limitation type for image resize. If min, images will be resized by limiting the minimum side length "
        "to `limit_side_len` (prior to accuracy). If max, images will be resized by limiting the maximum side "
        "length to `limit_side_len` (prior to speed). Default: max",
    )
    parser.add_argument(
        "--det_box_type",
        type=str,
        default="quad",
        choices=["quad", "poly"],
        help="box type for text region representation",
    )

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=str2bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="fast")

    # params for text recognizer
    parser.add_argument(
        "--rec_algorithm",
        type=str,
        default="CRNN",
        choices=["CRNN", "RARE", "CRNN_CH", "RARE_CH", "SVTR", "SVTR_PPOCRv3_CH"],
        help="recognition algorithm",
    )
    parser.add_argument(
        "--rec_amp_level",
        type=str,
        default="O0",
        choices=["O0", "O1", "O2", "O3"],
        help="Auto Mixed Precision level. This setting only works on GPU and Ascend",
    )
    parser.add_argument(
        "--rec_model_dir",
        type=str,
        help="directory containing the recognition model checkpoint best.ckpt, or path to a specific checkpoint file.",
    )  # determine the network weights
    # parser.add_argument("--rec_image_inverse", type=str2bool, default=True)
    parser.add_argument(
        "--rec_image_shape",
        type=str,
        default="3, 32, 320",
        help="C, H, W for target image shape. max_wh_ratio=W/H will be used to control the maximum width after "
        '"aspect-ratio-kept" resizing. Set W larger for longer text.',
    )

    parser.add_argument(
        "--rec_batch_mode",
        type=str2bool,
        default=True,
        help="Whether to run recognition inference in batch-mode, which is faster but may degrade the accuracy "
        "due to padding or resizing to the same shape.",
    )  # added
    parser.add_argument("--rec_batch_num", type=int, default=8)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default=None,
        help="path to character dictionary. If None, will pick according to rec_algorithm and red_model_dir.",
    )
    # uncomment it after model trained supporting space recognition.
    # parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument("--vis_font_path", type=str, default="docs/fonts/simfang.ttf")
    parser.add_argument("--drop_score", type=float, default=0.5)

    parser.add_argument(
        "--draw_img_save_dir",
        type=str,
        default="./inference_results",
        help="Dir to save visualization and detection/recognition/system prediction results",
    )
    parser.add_argument(
        "--save_crop_res",
        type=str2bool,
        default=False,
        help="Whether to save images cropped from text detection results.",
    )
    parser.add_argument(
        "--crop_res_save_dir", type=str, default="./output", help="Dir to save the cropped images for text boxes"
    )
    parser.add_argument(
        "--visualize_output",
        type=str2bool,
        default=False,
        help="Whether to visualize results and save the visualized image.",
    )

    parser.add_argument("--warmup", type=str2bool, default=False)
    parser.add_argument("--ocr_result_dir", type=str, default=None, help="path or directory of ocr results")
    parser.add_argument(
        "--ser_algorithm",
        type=str,
        default="VI_LAYOUTXLM",
        choices=["VI_LAYOUTXLM", "LAYOUTXLM"],
        help="ser algorithm",
    )
    parser.add_argument(
        "--ser_model_dir",
        type=str,
        help="directory containing the ser model checkpoint best.ckpt, or path to a specific checkpoint file.",
    )
    parser.add_argument(
        "--kie_batch_mode",
        type=str2bool,
        default=True,
        help="Whether to run recognition inference in batch-mode, which is faster but may degrade the accuracy "
        "due to padding or resizing to the same shape.",
    )
    parser.add_argument("--kie_batch_num", type=int, default=8)

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="",
        help="YAML config file specifying default arguments (default=" ")",
    )

    parser.add_argument(
        "--table_algorithm",
        type=str,
        default="TABLE_MASTER",
        choices=["TABLE_MASTER"],
        help="table structure recognition algorithm",
    )
    parser.add_argument(
        "--table_model_dir",
        type=str,
        help="directory containing the table structure recognition model checkpoint best.ckpt, "
        "or path to a specific checkpoint file.",
    )
    parser.add_argument(
        "--table_amp_level",
        type=str,
        default="O2",
        choices=["O0", "O1", "O2", "O3"],
        help="Auto Mixed Precision level. This setting only works on GPU and Ascend",
    )
    parser.add_argument(
        "--table_char_dict_path",
        type=str,
        default="./mindocr/utils/dict/table_master_structure_dict.txt",
        help="path to character dictionary for table structure recognition. "
        "If None, will pick according to table_algorithm and table_model_dir.",
    )
    parser.add_argument(
        "--table_max_len", type=int, default=480, help="max length of the input image for table structure recognition."
    )

    parser.add_argument(
        "--layout_algorithm",
        type=str,
        default="YOLOv8",
        choices=["YOLOv8", "LAYOUTLMV3"],
        help="layout analyzer algorithm",
    )
    parser.add_argument(
        "--layout_model_dir",
        type=str,
        help="directory containing the layout model checkpoint best.ckpt, or path to a specific checkpoint file.",
    )  # determine the network weights
    parser.add_argument(
        "--layout_category_dict_path",
        type=str,
        default="./mindocr/utils/dict/layout_category_dict.txt",
        help="path to category dictionary for layout recognition. "
        "If None, will pick according to layout_algorithm and layout_model_dir.",
    )
    parser.add_argument(
        "--layout_amp_level",
        type=str,
        default="O0",
        choices=["O0", "O1", "O2", "O3"],
        help="Auto Mixed Precision level. This setting only works on GPU and Ascend",
    )

    return parser


def parse_args():
    parser = create_parser()
    args = parser.parse_args()
    return args
