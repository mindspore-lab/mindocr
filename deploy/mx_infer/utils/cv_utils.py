from bisect import bisect_right
from math import floor

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

from .logger import logger_instance as log
from .safe_utils import safe_div


def get_hw_of_img(image_src: np.array):
    """
    get the hw of img
    :param image_src:
    :return:
    """
    if len(image_src.shape) > 2:
        # gbr/rgb
        height, width, _ = image_src.shape
    elif len(image_src.shape) == 2:
        # gray
        height, width = image_src.shape
    else:
        raise TypeError('image_src is not a image of gbr/gray')
    return height, width


def get_matched_gear_hw(image_hw: tuple, gear_list: list, max_dot_gear: tuple):
    """
    if find the matched gear, return tuple of (height,width)
    if not find the gear, return max_dot_gear
    :param image_hw:
    :param gear_list:
    :param max_dot_gear:
    :return:
    """
    index = len(gear_list)
    for i, (height, width) in enumerate(gear_list):
        if height >= image_hw[0] and width >= image_hw[1]:
            index = i
            break
    if index == len(gear_list):
        return max_dot_gear
    return gear_list[index]


def normalize(image_src: np.ndarray, scale, std, mean):
    """
    normalize by scale, mean, std. (x*scale - mean)/std
    :param image_src:
    :param scale: ndarray/int/float
    :param std: ndarray/int/float
    :param mean: ndarray/int/float
    :return:
    """
    image_dst = image_src.astype(np.float32)
    image_dst = safe_div((image_dst * scale - mean), std)
    return image_dst


def padding_with_cv(image_src: np.array, gear: tuple):
    """
    using open cv to padding the image.
    :param image_src:
    :param gear:
    :return:
    """
    height, width = get_hw_of_img(image_src=image_src)
    gear_h, gear_w = gear
    padding_h, padding_w = gear_h - height, gear_w - width
    image_dst = cv2.copyMakeBorder(image_src, 0, padding_h, 0, padding_w, cv2.BORDER_CONSTANT, value=0.)
    return image_dst


def to_chw_image(image_src: np.ndarray, as_contiguous: bool = True):
    """
    transpose the image from (height,width,channel) to (channel,height,width) format.
    using contiguous memory to speed up infer process
    :param image_src:
    :param as_contiguous:
    :return:
    """
    image_dst = image_src.transpose((2, 0, 1))
    if as_contiguous:
        return np.ascontiguousarray(image_dst)
    else:
        return image_dst


def expand(input_images):
    """
    if the type of input_images is numpy array,expand the first axis
    if the type of input_images is list, convert the list to numpy array
    :param input_images: input image
    :return: the numpy array of shape (batchsize,channel,height,width)
    """
    if isinstance(input_images, np.ndarray):
        input_array = np.expand_dims(input_images, 0).astype(np.float32)
    else:
        input_array = np.array(input_images).astype(np.float32)

    return input_array


def unclip(box: np.ndarray, unclip_ratio: float):
    """
    expand the box by unclip ratio
    """
    poly = Polygon(box)
    distance = safe_div(poly.area * unclip_ratio, poly.length)
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance)).reshape(-1, 1, 2)
    return expanded


def construct_box(box: np.ndarray, height: int, width: int, dest_height: int, dest_width: int):
    """
    resize the box to the original size.
    """
    try:
        box[:, 0] = np.clip(
            np.round(box[:, 0] / width * dest_width), 0, dest_width)
    except ZeroDivisionError as error:
        log.info(error)
    try:
        box[:, 1] = np.clip(
            np.round(box[:, 1] / height * dest_height), 0, dest_height)
    except ZeroDivisionError as error:
        log.info(error)

    return box.astype(np.int16)


def get_mini_boxes(contour):
    """
    get the box from the contours and make the points of box orderly.
    :param contour:
    :return:
    """
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    if points[1][1] > points[0][1]:
        index_one = 0
        index_four = 1
    else:
        index_one = 1
        index_four = 0
    if points[3][1] > points[2][1]:
        index_two = 2
        index_three = 3
    else:
        index_two = 3
        index_three = 2

    box = [points[index_one], points[index_two],
           points[index_three], points[index_four]]
    return np.array(box), min(bounding_box[1])


def box_score_fast(shrink_map: np.ndarray, input_box: np.ndarray):
    """
    using box mean score as the mean score
    :param shrink_map: the output feature map of DBNet
    :param input_box: the min boxes
    :return:
    """
    height, width = shrink_map.shape[:2]
    box = input_box.copy()
    x_min = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, width - 1)
    x_max = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, width - 1)
    y_min = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, height - 1)
    y_max = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, height - 1)

    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - x_min
    box[:, 1] = box[:, 1] - y_min
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(shrink_map[y_min:y_max + 1, x_min:x_max + 1], mask)[0]


def box_score_slow(shrink_map: np.ndarray, contour: np.ndarray):
    """
    using polyon mean score as the mean score
    :param shrink_map: the output feature map of DBNet
    :param contour: the contours
    :return:
    """
    height, width = shrink_map.shape
    contour = contour.copy()
    contour = np.reshape(contour, (-1, 2))

    xmin = np.clip(np.min(contour[:, 0]), 0, width - 1)
    xmax = np.clip(np.max(contour[:, 0]), 0, width - 1)
    ymin = np.clip(np.min(contour[:, 1]), 0, height - 1)
    ymax = np.clip(np.max(contour[:, 1]), 0, height - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

    contour[:, 0] = contour[:, 0] - xmin
    contour[:, 1] = contour[:, 1] - ymin

    cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(shrink_map[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def get_rotate_crop_image(img: np.ndarray, points: np.ndarray):
    """
    warp perspective an area into rectangle from img.
    :param img:
    :param points:
    :return: the sub img
    """
    if points.shape != (4, 2):
        raise ValueError("shape of points must be 4*2")
    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]),
                             np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]),
                              np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    m = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        m, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if safe_div(dst_img_height, dst_img_width) >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def get_batch_list_greedy(image_list_size: int, batchsize_list: list):
    max_batchsize_list = batchsize_list[-1]

    batch_list = [max_batchsize_list] * floor(safe_div(image_list_size, max_batchsize_list))
    image_list_size -= sum(batch_list)
    if image_list_size:
        batch_list.append(batchsize_list[bisect_right(batchsize_list, image_list_size)])
    return batch_list


def padding_batch(src_array: np.ndarray, batchsize: int):
    sample_size, channel, height, width = src_array.shape
    dst_array = src_array
    if batchsize - sample_size:
        zeros = np.zeros((batchsize - sample_size, channel, height, width)).astype(np.float32)
        dst_array = np.concatenate((src_array, zeros))
    return dst_array


def bgr_to_gray(src_image: np.ndarray):
    if len(src_image.shape) == 2:
        return src_image
    return cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)


def array_to_texts(output_array: np.ndarray, labels, actual_size: int = None):
    batchsize, length = output_array.shape
    if actual_size is not None:
        batchsize = actual_size
    texts = []
    for index in range(batchsize):
        char_list = []
        for i in range(length):
            if output_array[index, i] and i and output_array[index, i - 1] != output_array[index, i]:
                char_list.append(labels[output_array[index, i]])
        text = ''.join(char_list)
        texts.append(text)
    return texts


def resize_by_limit_max_side(src_image: np.ndarray, limit_side: int = 960):
    ratio = 1
    height, width = get_hw_of_img(src_image)
    if max(height, width) > limit_side:
        if height > width:
            ratio = safe_div(limit_side, height)
        else:
            ratio = safe_div(limit_side, width)

    resize_h = int(height * ratio)
    resize_w = int(width * ratio)

    resize_h = max(int(round(safe_div(resize_h, 32)) * 32), 32)
    resize_w = max(int(round(safe_div(resize_w, 32)) * 32), 32)
    dst_image = cv2.resize(src_image, (int(resize_w), int(resize_h)))
    return dst_image


def padding_with_np(input_tensor: np.ndarray, gear: tuple, nchw: bool = True):
    if nchw:
        batchsize, channel, height, width = input_tensor.shape
    else:
        batchsize, height, width, channel = input_tensor.shape
    padding_tensor = np.zeros((batchsize, channel, gear[0], gear[1]), dtype=np.float32)
    padding_tensor[:, :, :height, :width] = input_tensor
    return padding_tensor
