from typing import Union

import numpy as np

_2PI = 2 * np.pi


def sort_clockwise(polygon: Union[list, np.ndarray]) -> np.ndarray:
    """
    Sort polygon (must be a convex) vertices in clockwise order (origin is top-left corner).
    Args:
        polygon:  numpy array with any number of vertices (N, 2).
    Returns:
        polygon with vertices sorted in clockwise order.
    """
    if isinstance(polygon, list):
        polygon = np.array(polygon)

    center = polygon.mean(0)
    angles = (np.arctan2(*(polygon - center).T[::-1]) + np.pi) % _2PI
    return polygon[np.argsort(angles)]
