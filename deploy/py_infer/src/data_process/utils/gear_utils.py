from bisect import bisect_right
from math import floor
from typing import Dict, List, Tuple

import numpy as np

from ...utils import safe_div


def get_matched_gear_hw(image_hw: Tuple[int], hw_list: Tuple[Tuple[int]]) -> Tuple[int]:
    """
    find the most matched gear
    """
    if len(hw_list) == 1:
        return hw_list[0]

    origin_h, origin_w = image_hw[0], image_hw[1]
    matched_shape = None
    min_diff = float("inf")
    for i, (height, width) in enumerate(hw_list):
        dist = abs(height - origin_h) + abs(width - origin_w)
        if dist < min_diff:
            min_diff = dist
            matched_shape = hw_list[i]
    return matched_shape


def get_matched_gear_bs(image_num: int, bs_list: Tuple[int]) -> Tuple[int]:
    max_bs_list: int = bs_list[-1]

    batch_list = [max_bs_list] * floor(safe_div(image_num, max_bs_list))
    image_num -= sum(batch_list)
    if image_num:
        batch_list.append(bs_list[bisect_right(bs_list, image_num)])
    return tuple(batch_list)


def padding_to_batch(data: Dict, bs: int) -> Dict:
    """
    padding the batch size of data["net_inputs"] to bs
    """

    def _padding_to_batch_np(x: np.ndarray, bs) -> np.ndarray:
        sample_size, *other_shape = x.shape
        y = x
        if bs - sample_size:
            zeros = np.zeros((bs - sample_size, *other_shape)).astype(x.dtype)
            y = np.concatenate((x, zeros))
        return y

    net_inputs = data["net_inputs"]
    data["net_inputs"] = [_padding_to_batch_np(x, bs) for x in net_inputs]

    return data


def get_batch_from_padding(input: List[np.ndarray], batch: int = -1) -> List[np.ndarray]:
    if batch >= 0:
        input = [x[:batch, ...] if x.shape[0] > batch else x for x in input]

    return input


def split_by_size(input: List, size: List[int]):
    start_index = 0
    for batch in size:
        yield input[start_index : start_index + batch]
        start_index += batch
