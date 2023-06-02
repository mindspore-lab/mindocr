from bisect import bisect_right
from typing import Union, Dict, List

import numpy as np
from math import floor

from ...utils import safe_div


def get_matched_gear_hw(image_hw: tuple, hw_list: list):
    """
    hw_list is sorted by h*w.
    if find the matched gear, return tuple of (h,w)
    if not find the gear, return hw_list[-1]
    """
    index = len(hw_list)
    for i, (height, width) in enumerate(hw_list):
        if height >= image_hw[0] and width >= image_hw[1]:
            index = i
            break
    if index == len(hw_list):
        return hw_list[-1]

    return hw_list[index]


def get_matched_gear_bs(image_num: int, bs_list: list):
    max_bs_list = bs_list[-1]

    batch_list = [max_bs_list] * floor(safe_div(image_num, max_bs_list))
    image_num -= sum(batch_list)
    if image_num:
        batch_list.append(bs_list[bisect_right(bs_list, image_num)])
    return batch_list


def padding_to_batch(input: Union[np.ndarray, Dict], bs: int):
    image = input["image"] if isinstance(input, dict) else input

    sample_size, channel, height, width = image.shape
    output = image
    if bs - sample_size:
        zeros = np.zeros((bs - sample_size, channel, height, width)).astype(image.dtype)
        output = np.concatenate((image, zeros))

    if isinstance(input, dict):
        output = {**input, "image": output}

    return output


def get_batch_from_padding(input: Union[np.ndarray, List], batch: int):
    if batch is not None:
        input = input[:batch, ...] if isinstance(input, np.ndarray) else [x[:batch, ...] for x in input]

    return input


def split_by_size(input: list, size: list):
    start_index = 0
    for batch in size:
        yield input[start_index: start_index + batch]
        start_index += batch
