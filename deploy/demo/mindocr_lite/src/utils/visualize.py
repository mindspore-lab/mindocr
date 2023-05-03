"""
OCR visualization methods
"""
import os.path
from typing import Union, List
from enum import Enum, unique

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw


@unique
class VisMode(Enum):
    bbox = 0
    bbox_text = 1
    crop = 2


class Visualization(object):
    def __init__(self, vis_mode: VisMode):
        """
        Class for visualization of detection and recognition results.
        Args:
            vis_mode: Input values can be VisMode.bbox, VisMode.bbox_text, VisMode.crop.
                VisMode.bbox: visualize the detected bounding boxes on the image
                VisMode.bbox_text: visualize the detected bounding boxes and the recognized texts on the image
                VisMode.crop: crop the image patches according to detected bounding boxes and visualize
        """
        self.vis_mode = vis_mode

    def vis_bbox(self, image, box_list, color, thickness):
        image = image.copy()
        for box in box_list:
            box = box.astype(int)
            cv2.polylines(image, [box], True, color, thickness)
        return image

    def vis_bbox_text(self, image, box_list, text_list, color, thickness, font_path):
        image_bbox = self.vis_bbox(image, box_list, color, thickness)
        image_text = image_bbox.copy()
        image_text.fill(255)
        image_text = self.vis_bbox(image_text, box_list, color, thickness)

        image_text = Image.fromarray(image_text)
        draw_text = ImageDraw.Draw(image_text)
        font = ImageFont.truetype(font_path, 20, encoding='utf-8')
        for i, text in enumerate(text_list):
            draw_text.text(box_list[i][0], text, color, font)
        image_concat = np.concatenate([np.array(image_bbox), np.array(image_text)], axis=1)
        return image_concat

    def vis_crop(self, image, box_list):
        image_crop = []
        for box in box_list:
            if box.shape != (4, 2):
                raise ValueError("shape of crop box must be 4*2")
            box = box.astype(np.float32)
            img_crop_width = int(max(np.linalg.norm(box[0] - box[1]),
                                     np.linalg.norm(box[2] - box[3])))
            img_crop_height = int(max(np.linalg.norm(box[0] - box[3]),
                                      np.linalg.norm(box[1] - box[2])))
            pts_std = np.float32([[0, 0], [img_crop_width, 0],
                                  [img_crop_width, img_crop_height],
                                  [0, img_crop_height]])
            m = cv2.getPerspectiveTransform(box, pts_std)
            dst_img = cv2.warpPerspective(
                image,
                m, (img_crop_width, img_crop_height),
                borderMode=cv2.BORDER_REPLICATE,
                flags=cv2.INTER_CUBIC)
            dst_img_height, dst_img_width = dst_img.shape[0:2]
            if dst_img_width != 0  and dst_img_height / dst_img_width >= 1.5:
                dst_img = np.rot90(dst_img)
            image_crop.append(dst_img)
        return image_crop

    def __call__(self,
                 image: np.array,
                 box_list: List[np.array],
                 text_list: List[str] = None,
                 color: tuple = (0, 0, 255),
                 thickness: int = 2,
                 font_path: str = None) -> Union[List[np.array], np.array]:
        """
        Args:
            image: single input image
            box_list: list of detected bounding boxes
            text_list: list of recognized texts
            color: color of font and bounding box
            thickness: thickness of bounding box
            font_path: path of font file

        Returns: an image or list of cropped images
        """
        if self.vis_mode == VisMode.bbox:
            return self.vis_bbox(image, box_list, color, thickness)
        elif self.vis_mode == VisMode.bbox_text:
            if not font_path:
                font_path = os.path.join(os.path.dirname(__file__), '../asset/font/simfang.ttf')
            return self.vis_bbox_text(image, box_list, text_list, color, thickness, font_path)
        elif self.vis_mode == VisMode.crop:
            return self.vis_crop(image, box_list)
        else:
            raise TypeError("Invalid 'vis_mode'. The type of 'vis_mode' should be VisMode.")
