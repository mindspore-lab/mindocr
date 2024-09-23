from typing import List

import numpy as np
from shapely.geometry import Polygon


def check_polygons_valid(polygons: List[Polygon]) -> bool:
    return all([polygon.is_valid for polygon in polygons])


def calculate_iou(box1: np.array, box2: np.array) -> float:
    # convert np.array to Polygon
    poly1 = Polygon(box1[:8].reshape((4, 2)))
    poly2 = Polygon(box2[:8].reshape((4, 2)))
    # if any of boxes is invalid, return False
    if not check_polygons_valid([poly1, poly2]):
        return 0
    inter_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    # if union is 0, return 0
    if union_area == 0:
        return 0
    return inter_area / union_area


def should_merge(box1: np.array, box2: np.array, threshold: float) -> bool:
    return calculate_iou(box1, box2) > threshold


def weighted_merge(box1: np.array, box2: np.array) -> np.array:
    merged_box = np.zeros(9)
    merged_box[:8] = (box1[8] * box1[:8] + box2[8] * box2[:8]) / (box1[8] + box2[8])
    merged_box[8] = box1[8] + box2[8]
    return merged_box


def standard_nms(boxes: List[np.array], threshold: float) -> np.array:
    # initial filtered boxes list
    filtered_boxes = []
    sorted_boxes = sorted(boxes, key=lambda x: x[8], reverse=True)
    while sorted_boxes:
        max_score_box = sorted_boxes.pop(0)
        filtered_boxes.append(max_score_box)
        sorted_boxes = list(filter(lambda x: calculate_iou(max_score_box, x) < threshold, sorted_boxes))
    return np.array(filtered_boxes)


def merge_quadrangle_n9(geometries: np.array, threshold: float = 0.3) -> np.array:
    """
    locality-aware NMS for Windows version
    :param: geometries: a numpy array with shape (N,9)
    :param: threshold: IOU threshold
    :return: filtered bounding boxes
    """
    s = []
    p = None
    for g in geometries:
        if (p is not None) and should_merge(g, p, threshold):
            p = weighted_merge(g, p)
        else:
            if p is not None:
                s.append(p)
            p = g
    if p is not None:
        s.append(p)
    return standard_nms(s, threshold)
