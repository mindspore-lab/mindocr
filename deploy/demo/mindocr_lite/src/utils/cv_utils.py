from bisect import bisect_right
from math import floor

import cv2
import numpy as np

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


def padding_with_np(input_tensor: np.ndarray, gear: tuple, nchw: bool = True):
    if nchw:
        batchsize, channel, height, width = input_tensor.shape
    else:
        batchsize, height, width, channel = input_tensor.shape
    padding_tensor = np.zeros((batchsize, channel, gear[0], gear[1]), dtype=np.float32)
    padding_tensor[:, :, :height, :width] = input_tensor
    return padding_tensor
