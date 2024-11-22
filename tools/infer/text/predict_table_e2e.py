"""
Infer end-to-end from images and convert them into docx.

Example:
    $ python tools/infer/text/predict_table_e2e.py --image_dir {path_to_img}
"""
import json
import logging
import os
import time
from typing import List

import cv2
from config import create_parser, str2bool
from predict_layout import LayoutAnalyzer
from predict_system import TextSystem
from predict_table_recognition import TableAnalyzer
from utils import (
    add_padding,
    convert_info_docx,
    get_dict_from_file,
    get_image_paths,
    sort_words_by_poly,
    sorted_layout_boxes,
)

logger = logging.getLogger("mindocr")


def e2e_parse_args():
    """
    Inherit the parser from the config.py file, and add the following arguments:
        1. layout: Whether to enable layout analyzer
        2. ocr: Whether to enable ocr
        3. table: Whether to enable table recognizer
        4. recovery: Whether to recovery output to docx
    """
    parser = create_parser()

    parser.add_argument(
        "--layout",
        type=str2bool,
        default=True,
        help="Whether to enable layout analyzer. The default layout analysis algorithm is YOLOv8.",
    )

    parser.add_argument(
        "--ocr",
        type=str2bool,
        default=True,
        help="Whether to enable ocr. The default ocr detection algorithm is DB++ and recognition algorithm is CRNN.",
    )

    parser.add_argument(
        "--table",
        type=str2bool,
        default=True,
        help="Whether to table recognizer. The default table analysis algorithm is TableMaster.",
    )

    parser.add_argument(
        "--recovery",
        type=str2bool,
        default=True,
        help="Whether to recovery output to docx. The docx will be saved in the ./inferrence_results as default.",
    )

    args = parser.parse_args()
    return args


def init_ocr(args):
    """
    Initialize text detection and recognition system

    Args:
        ocr: enable text system or not
        det_algorithm: detection algorithm
        rec_algorithm: recognition algorithm
        det_model_dir: detection model directory
        rec_model_dir: recognition model directory
    """
    if args.ocr:
        return TextSystem(args)

    return None


def init_layout(args):
    """
    Initialize layout analysis system

    Args:
        layout: enable layout module or not
        layout_algorithm: layout algorithm
        layout_model_dir: layout model ckpt path
        layout_amp_level: Auto Mixed Precision level for layout
    """
    if args.layout:
        return LayoutAnalyzer(args)

    return None


def init_table(args):
    """
    Initialize table recognition system

    Args:
        table: enable table recognizer or not
        table_algorithm: table algorithm
        table_model_dir: table model ckpt path
        table_max_len: max length of the input image
        table_char_dict_path: path to character dictionary for table
        table_amp_level: Auto Mixed Precision level for table
    """
    if args.table:
        return TableAnalyzer(args)

    return None


def save_e2e_res(e2e_res: List, img_path: str, save_path: str):
    """
    Save the end-to-end results to a txt file
    """
    lines = []
    img_name = os.path.basename(img_path).rsplit(".", 1)[0]
    save_path = os.path.join(save_path, img_name + "_e2e_result.txt")
    for i, res in enumerate(e2e_res):
        img_pred = str(json.dumps(res)) + "\n"
        lines.append(img_pred)

    with open(save_path, "w") as f:
        f.writelines(lines)
        f.close()


def predict_table_e2e(
    img_path, layout_category_dict, layout_analyzer, text_system, table_analyzer, do_visualize, save_folder, recovery
):
    """
    Predict the end-to-end results for the input image

    Args:
        img_path: path to the input image
        layout_category_dict: category dictionary for layout recognition
        layout_analyzer: layout analyzer model, for more details, please refer to predict_layout.py
        text_system: text system model, for more details, please refer to predict_system.py
        table_analyzer: table analyzer model, for more details, please refer to predict_table.py
        do_visualize: whether to visualize the output
        save_folder: folder to save the output
        recovery: whether to recovery the output to docx
    """
    img_name = os.path.basename(img_path).rsplit(".", 1)[0]
    image = cv2.imread(img_path)

    if text_system is not None and do_visualize:
        text_system(img_path, do_visualize=do_visualize)

    if layout_analyzer is not None:
        results = layout_analyzer(img_path, do_visualize=do_visualize)
    else:
        results = [{"category_id": 1, "bbox": [0, 0, image.shape[1], image.shape[0]], "score": 1.0}]

    logger.info(f"Infering {len(results)} detected regions in {img_path}")

    # crop text regions
    h_ori, w_ori = image.shape[:2]
    final_results = []
    for i in range(len(results)):
        category_id = results[i]["category_id"]
        left, top, w, h = results[i]["bbox"]
        right = left + w
        bottom = top + h
        cropped_img = image[int(top) : int(bottom), int(left) : int(right)]

        if (category_id == 1 or category_id == 2 or category_id == 3) and text_system is not None:
            start_time = time.time()

            # only add white padding for text, title and list images for better recognition
            if layout_analyzer is not None:
                cropped_img = add_padding(cropped_img, padding_size=10, padding_color=(255, 255, 255))

            rec_res_all_crops = text_system(cropped_img, do_visualize=False)
            output = sort_words_by_poly(rec_res_all_crops[1], rec_res_all_crops[0])
            final_results.append(
                {"type": layout_category_dict[category_id], "bbox": [left, top, right, bottom], "res": " ".join(output)}
            )

            logger.info(
                f"Processing {layout_category_dict[category_id]} at [{left}, {top}, {right}, {bottom}]"
                f" {time.time() - start_time:.2f}s"
            )
        elif category_id == 4 and table_analyzer is not None:
            start_time = time.time()
            pred_html, _ = table_analyzer(cropped_img, do_visualize=do_visualize)
            final_results.append(
                {"type": layout_category_dict[category_id], "bbox": [left, top, right, bottom], "res": pred_html}
            )

            logger.info(
                f"Processing {layout_category_dict[category_id]} at [{left}, {top}, {right}, {bottom}]"
                f" {time.time() - start_time:.2f}s"
            )
        else:
            start_time = time.time()
            save_path = os.path.join(save_folder, f"{img_name}_figure_{i}.png")
            cv2.imwrite(save_path, cropped_img)
            final_results.append(
                {"type": layout_category_dict[category_id], "bbox": [left, top, right, bottom], "res": save_path}
            )

            logger.info(
                f"Processing {layout_category_dict[category_id]} at [{left}, {top}, {right}, {bottom}]"
                f" {time.time() - start_time:.2f}s"
            )

    if recovery:
        final_results = sorted_layout_boxes(final_results, w_ori)
        convert_info_docx(final_results, save_folder, f"{img_name}_converted_docx")

    return final_results


def main():
    from mindocr.utils.logger import set_logger

    set_logger(name="mindocr")

    first_time = time.time()
    args = e2e_parse_args()
    save_folder = args.draw_img_save_dir

    save_folder, _ = os.path.splitext(save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    text_system = init_ocr(args)
    layout_analyzer = init_layout(args)
    layout_category_dict = get_dict_from_file(args.layout_category_dict_path)
    table_analyzer = init_table(args)

    img_paths = get_image_paths(args.image_dir)
    for i, img_path in enumerate(img_paths):
        logger.info(f"Infering [{i+1}/{len(img_paths)}]: {img_path}")
        final_results = predict_table_e2e(
            img_path,
            layout_category_dict,
            layout_analyzer,
            text_system,
            table_analyzer,
            args.visualize_output,
            save_folder,
            args.recovery,
        )

        save_e2e_res(final_results, img_path, save_folder)

    logger.info(f"Processing e2e total time: {time.time() - first_time:.2f}s")
    logger.info(f"Done! predict {len(img_paths)} e2e results saved in {save_folder}")


if __name__ == "__main__":
    main()
