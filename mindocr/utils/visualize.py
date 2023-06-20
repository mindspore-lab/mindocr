import math
from typing import List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .logger import Logger

__all__ = ["show_img", "show_imgs", "draw_boxes", "draw_texts_with_boxes", "recover_image", "visualize"]
_logger = Logger("mindocr")


def show_img(img: np.array, is_bgr_img=True, title="img", show=True, save_path=None):
    color = len(img.shape) == 3 and img.shape[-1] == 3
    if is_bgr_img:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # imgs = np.expand_dims(imgs, axis=0)
    plt.figure()
    plt.title("{}_{}".format(title, 0))
    plt.imshow(img, cmap=None if color else "gray")
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)  # , bbox_inches='tight', dpi=460)


def show_imgs(
    imgs: List[np.array],
    is_bgr_img=False,
    title=None,
    show=True,
    save_path=None,
    mean_rgb=None,
    std_rgb=None,
    is_chw=False,
    figure_width=6.4,
):
    # if len(imgs.shape) not in [2, 4]:
    #    imgs = np.expand_dims(imgs, axis=0)
    subplot_h = 4.8 * (figure_width / 6.4)
    figure_height = len(imgs) * subplot_h
    plt.figure(figsize=(figure_width, figure_height))
    num_images = len(imgs)  # imgs.shape[0]
    for i, _img in enumerate(imgs):
        img = _img.copy()
        if is_chw:
            img = img.transpose([1, 2, 0])
        color = len(img.shape) == 3 and img.shape[-1] == 3
        if is_bgr_img:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if mean_rgb and std_rgb:
            img = (img * std_rgb) + mean_rgb
            img = np.clip(img, 0, 255).astype(np.int32)
        plt.subplot(num_images, 1, i + 1)
        if title is not None:
            plt.title("{}_{}".format(title, i))
        plt.imshow(img, cmap=None if color else "gray")
    if show:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)


def draw_boxes(
    image: Union[str, np.array],
    bboxes: Union[list, np.array],
    color: Union[tuple, str] = (255, 0, 0),
    thickness=1,
    is_bgr_img=False,
    color_random=False,
):  # , to_rgb=False):
    """image can be str or np.array for image in 'BGR' colorm mode.
    color: list for color of each box, or tuple for color of all boxes with the same color. in RGB order

    return:
        np.array, image draw with boxes in RGB color order
    """
    # load image and convert to BGR format
    if isinstance(image, str):
        image = cv2.imread(image)
    else:
        image = image.copy()
        if not is_bgr_img:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for i, box in enumerate(bboxes):
        box = box.astype(int)

        if isinstance(color, tuple):
            color_bgr = color[::-1]
        else:
            color_bgr = np.randint(0, 255, 3, dtype=np.int32)
        cv2.polylines(image, [box], True, color_bgr, thickness)

    # convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def draw_texts_with_boxes(
    image: Union[str, np.array],
    bboxes: Union[list, np.array],
    texts: List[str],
    box_color: Union[tuple, str] = (255, 0, 0),
    thickness=1,
    text_color: tuple = (0, 0, 0),
    font_path: str = None,
    font_size: int = None,
    is_bgr_img: bool = False,
    hide_boxes: bool = False,
    text_inside_box: bool = True,
):
    """
    font_path: path to ttf font file. If None, use a "better than nothing" defalut font in PIL (font_size can't be set
         in this case)
    font_size: font size. if None, font_size will be computed automatically according to the box size and image size.
    """
    if hide_boxes:
        if is_bgr_img:
            image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        pimg = Image.fromarray(image)
    else:
        img_with_boxes = draw_boxes(image, bboxes, box_color, thickness=thickness, is_bgr_img=is_bgr_img)
        pimg = Image.fromarray(img_with_boxes)

    img_h, img_w = pimg.size

    img_draw = ImageDraw.Draw(pimg)

    def _get_draw_point_and_font_size(box, font_size="auto", text_inside_box=True, img_h=736):
        pt_sums = np.array(box).sum(axis=1)
        corner = box[np.argmin(pt_sums)]

        # box_h = box[:, 1].max() - box[:, 1].min()
        # box_w = box[:, 0].max() - box[:, 0].min()

        box_h = int(math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2))
        box_w = int(math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2))

        # TODO: consider the height and witdh of the text
        if text_inside_box:
            draw_point_w = corner[0] + box_w * 0.1
            draw_point_h = corner[1] - box_h * 0.05
            font_size = round(box_h * 0.9) if not isinstance(font_size, int) else font_size
        else:
            if isinstance(font_size, int) or isinstance(font_size, float):
                font_size = font_size
            else:
                _logger.warning("font size needs to be fixed if text placed under box")
                font_size = 20  # round(img_h * 0.05)
            draw_point_w = corner[0]
            draw_point_h = corner[1] - font_size

        return (draw_point_w, draw_point_h), font_size

    for i, text in enumerate(texts):
        # draw text on the most left-top point
        box = bboxes[i]
        draw_point, fs = _get_draw_point_and_font_size(box, font_size, text_inside_box=text_inside_box, img_h=img_h)

        # TODO: use other lib which can set font size dynamically after font loading
        font = ImageFont.load_default() if not font_path else ImageFont.truetype(font_path, fs, encoding="utf-8")

        font_width = font.getsize(text)[0]
        box_width = int(math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2))
        if font_width > box_width:
            font_size = int(fs * box_width / font_width)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

        # refine the draw starting
        img_draw.text(draw_point, text, font=font, fill=text_color)

    return np.array(pimg)


def visualize(
    rgb_img, boxes, texts=None, vis_font_path=None, display=True, save_path=None, draw_texts_on_blank_page=True
):
    det_vis = draw_boxes(rgb_img, boxes)
    vis_imgs = [det_vis]
    if texts is not None:
        if draw_texts_on_blank_page:
            bg = np.ones([rgb_img.shape[0], rgb_img.shape[1], 3], dtype=np.uint8) * 255
            hide_boxes = False
            text_inside_box = True
            font_size = None
            text_color = (0, 0, 0)
        else:
            bg = det_vis
            hide_boxes = True
            text_inside_box = False
            font_size = max(int(18 * rgb_img.shape[0] / 700), 22)
            text_color = (0, 255, 0)

        text_vis = draw_texts_with_boxes(
            bg,
            boxes,
            texts,
            text_color=text_color,
            font_path=vis_font_path,
            font_size=font_size,
            hide_boxes=hide_boxes,
            text_inside_box=text_inside_box,
        )

        if draw_texts_on_blank_page:
            vis_imgs = [det_vis, text_vis]
        else:
            vis_imgs = [text_vis]

    show_imgs(vis_imgs, show=display, title=None, save_path=save_path)


def recover_image(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, is_chw=True, to_bgr=True):
    """
    recover normalized image for visualization
    img: must be in rgb mode"""
    if img.dtype == "uint8":
        return img

    if is_chw:
        img = img.transpose((1, 2, 0))

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    img = (img * std) + mean

    if to_bgr:
        img = img[..., [2, 1, 0]]
    img = img.astype(np.uint8)

    return img
