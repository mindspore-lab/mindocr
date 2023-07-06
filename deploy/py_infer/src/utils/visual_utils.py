"""
OCR visualization methods
"""
import math
import os.path
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = ["vis_bbox", "vis_bbox_text", "vis_crop"]


def vis_bbox(image, box_list, color, thickness):
    """
    Draw a bounding box on an image.
    :param image: input image
    :param box_list: box list to add on image
    :param color: color of the box
    :param thickness: line thickness
    :return: image with box
    """

    image = image.copy()
    for box in box_list:
        box = box.astype(int)
        cv2.polylines(image, [box], True, color, thickness)
    return image


def vis_bbox_text(image, box_list, text_list, font_path):
    """
    Draw a bounding box and text on an image.
    :param image: input image
    :param box_list: box list to add on image
    :param text_list: text list to add on image
    :param font_path: path to font file
    :return: image with box and text
    """
    if font_path is None:
        _font_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../docs/fonts/simfang.ttf"))
        if os.path.isfile(_font_path):
            font_path = _font_path

    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
    random.seed(0)

    draw_left = ImageDraw.Draw(img_left)
    if text_list is None or len(text_list) != len(box_list):
        text_list = [None] * len(box_list)
    for idx, (box, txt) in enumerate(zip(box_list, text_list)):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw_left.polygon(box.astype(np.float32), fill=color)
        img_right_text = draw_box_txt_fine((w, h), box, txt, font_path)
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_right_text, [pts], True, color, 1)
        img_right = cv2.bitwise_and(img_right, img_right_text)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new(mode="RGB", size=(w * 2, h), color=(255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
    return np.array(img_show)


def draw_box_txt_fine(img_size, box, txt, font_path):
    box_height = int(math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2))
    box_width = int(math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2))
    img_text = Image.new("RGB", (box_width, box_height), (255, 255, 255))
    if box_height > 2 * box_width and box_height > 30:
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_height, box_width), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)
        img_text = img_text.transpose(Image.ROTATE_270)
    else:
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_width, box_height), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)
    pts1 = np.float32([[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]])
    pts2 = np.array(box, dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img_text = np.array(img_text, dtype=np.uint8)
    img_right_text = cv2.warpPerspective(
        img_text, M, img_size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
    return img_right_text


def create_font(txt, sz, font_path):
    font_size = int(sz[1] * 0.99)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    length = font.getsize(txt)[0]
    if length > sz[0]:
        font_size = int(font_size * sz[0] / length)
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    return font


def vis_crop(image, box_list):
    """
    Generate crop image
    :param image: input image
    :param box_list: list of box
    :return List of Cropped Images
    """
    image_crop = []
    for box in box_list:
        if box.shape != (4, 2):
            raise ValueError("shape of crop box must be 4*2")
        box = box.astype(np.float32)
        img_crop_width = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
        img_crop_height = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
        m = cv2.getPerspectiveTransform(box, pts_std)
        dst_img = cv2.warpPerspective(
            image, m, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
        )
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_width != 0 and dst_img_height / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        image_crop.append(dst_img)
    return image_crop
