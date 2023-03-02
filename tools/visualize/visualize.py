"""
OCR visualization methods
"""
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
        # image_text = Image.new('RGB', (image_bbox.width, image_bbox.height), (255, 255, 255))
        image_text = image_bbox.copy()
        # image_text = np.zeros((image_bbox.shape[0], image_bbox.shape[1], 3), np.uint8)
        image_text.fill(255)
        image_text = self.vis_bbox(image_text, box_list, color, thickness)

        # image_bbox = Image.fromarray(cv2.cvtColor(image_bbox, cv2.COLOR_BGR2RGB))
        image_text = Image.fromarray(image_text)
        draw_text = ImageDraw.Draw(image_text)
        font = ImageFont.truetype(font_path, 20, encoding='utf-8')
        for i, text in enumerate(text_list):
            # draw_text.polygon(box_list[i], fill='blue', outline='blue')
            draw_text.text(box_list[i][0], text, color, font)
        image_concat = np.concatenate([np.array(image_bbox), np.array(image_text)], axis=1)
        return image_concat

    def vis_crop(self, image, box_list):
        image = Image.fromarray(image)
        image_crop = []
        for box in box_list:
            crop = image.crop((box[0][0], box[0][1], box[2][0], box[2][1]))
            image_crop.append(crop)
        return image_crop

    def __call__(self,
                 image: np.array,
                 box_list: List[np.array],
                 text_list: List[str] = None,
                 color: tuple = (0, 0, 255),
                 thickness: int = 2,
                 font_path: str = './simfang.ttf') -> Union[List[np.array], np.array]:
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
            return self.vis_bbox_text(image, box_list, text_list, color, thickness, font_path)
        elif self.vis_mode == VisMode.crop:
            return self.vis_crop(image, box_list)
        else:
            raise TypeError("Invalid 'vis_mode'. The type of 'vis_mode' should be VisMode.")


if __name__ == '__main__':
    img_dir = './input_img.jpg'
    image = cv2.imread(img_dir)
    box_list = [np.array([[0, 0], [0, 100], [200, 100], [200, 0]]),
                np.array([[50, 50], [50, 300], [700, 300], [700, 50]])]
    text_list = ['text1', '中文2']

    # bbox
    visualization_bbox = Visualization(vis_mode=VisMode.bbox)
    image_bbox = visualization_bbox(image, box_list)
    print(image.shape)
    print(image_bbox.shape)
    cv2.imwrite('./img_bbox.jpg', image_bbox)

    # bbox and text
    visualization_bbox_text = Visualization(vis_mode=VisMode.bbox_text)
    image_bbox_text = visualization_bbox_text(image, box_list, text_list)
    print(image.shape)
    print(image_bbox_text.shape)
    cv2.imwrite('./img_bbox_text.jpg', image_bbox_text)

    # crop
    visualization_crop = Visualization(vis_mode=VisMode.crop)
    image_crop = visualization_crop(image, box_list)
    print(image.shape)
    for img in image_crop:
        print(img.size)
    for i, img in enumerate(image_crop):
        cv2.imwrite(f'./img_crop_{i}.jpg', np.ascontiguousarray(img))
