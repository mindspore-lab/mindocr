"""
Infer table from images with structure model and ocr model.

Example:
    $ python tools/infer/text/predict_table_recognition.py --image_dir {path_to_img} --table_algorithm TABLE_MASTER
"""
import logging
import os
import sys
from typing import Union

import cv2
import numpy as np
from config import parse_args
from predict_system import TextSystem
from predict_table_structure import StructureAnalyzer
from utils import TableMasterMatcher, get_image_paths

from mindocr.utils.logger import set_logger

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))

logger = logging.getLogger("mindocr")


class TableAnalyzer:
    """
    Model inference class for table structure analysis and match with ocr result.
    Example:
        >>> args = parse_args()
        >>> analyzer = TableAnalyzer(args)
        >>> img_path = "path/to/image.jpg"
        >>> pred_html, time_prof = analyzer(img_path)
    """

    def __init__(self, args):
        self.text_system = TextSystem(args)
        self.table_structure = StructureAnalyzer(args)
        self.match = TableMasterMatcher()

    def _structure(self, img_or_path: Union[str, np.ndarray], do_visualize: bool = True):
        structure_res, elapse = self.table_structure(img_or_path, do_visualize)
        return structure_res, elapse

    def _text_ocr(self, img_or_path: Union[str, np.ndarray], do_visualize: bool = True):
        boxes, text_scores, time_prof = self.text_system(img_or_path, do_visualize)
        if isinstance(img_or_path, str):
            img = cv2.imread(img_or_path)
        elif isinstance(img_or_path, np.ndarray):
            img = img_or_path
        else:
            raise ValueError("Invalid input type, should be str or np.ndarray.")
        h, w = img.shape[:2]
        r_boxes = []
        for box in boxes:
            x_min = max(0, box[:, 0].min() - 1)
            x_max = min(w, box[:, 0].max() + 1)
            y_min = max(0, box[:, 1].min() - 1)
            y_max = min(h, box[:, 1].max() + 1)
            box = [x_min, y_min, x_max, y_max]
            r_boxes.append(box)
        dt_boxes = np.array(r_boxes)
        return dt_boxes, text_scores, time_prof

    def __call__(self, img_or_path: Union[str, np.ndarray], do_visualize: bool = True):
        boxes, text_scores, ocr_time_prof = self._text_ocr(img_or_path, do_visualize)
        structure_res, struct_time_prof = self._structure(img_or_path, do_visualize)
        pred_html = self.match(structure_res, boxes, text_scores)
        time_prof = {
            "ocr": ocr_time_prof,
            "table": struct_time_prof,
        }
        return pred_html, time_prof


def parse_html_table(html_table):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_table, "html.parser")
    table = soup.find("table")
    if not table:
        raise ValueError("No table found in the HTML string.")
    return table


def to_excel(html_table, excel_path):
    from tablepyxl import tablepyxl

    table = parse_html_table(html_table)
    tablepyxl.document_to_xl(str(table), excel_path)


def to_csv(html_table, csv_path):
    import pandas as pd

    table = parse_html_table(html_table)
    df = pd.read_html(str(table))[0]
    df.to_csv(csv_path, index=False)


def main():
    args = parse_args()
    set_logger(name="mindocr")
    analyzer = TableAnalyzer(args)
    img_paths = get_image_paths(args.image_dir)
    save_dir = args.draw_img_save_dir
    for i, img_path in enumerate(img_paths):
        logger.info(f"Infering {i+1}/{len(img_paths)}: {img_path}")
        pred_html, time_prof = analyzer(img_path, do_visualize=True)
        logger.info(f"Time profile: {time_prof}")
        img_name = os.path.basename(img_path).rsplit(".", 1)[0]
        to_csv(pred_html, os.path.join(save_dir, f"{img_name}.csv"))
    logger.info(f"Done! All structure results are saved to {args.draw_img_save_dir}")


if __name__ == "__main__":
    main()
