import cv2
import numpy as np


def fill_hole(src_mask) -> np.array:
    """Fill holes in matrix"""
    src_mask = np.array(src_mask)
    h, w = src_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1 : h + 1, 1 : w + 1] = src_mask.copy()

    mask = np.zeros((h + 4, w + 4), np.uint8)

    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1 : h + 1, 1 : w + 1].astype(np.bool_)

    return ~canvas | src_mask
