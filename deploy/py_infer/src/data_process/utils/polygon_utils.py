from typing import List, Union

import numpy as np
from shapely.geometry import Polygon


def poly_iou(poly_a: Union[List, Polygon], poly_b: Union[List, Polygon], zero_division: Union[int, float] = 0) -> float:
    """Return the iou between two poly"""

    if not isinstance(poly_a, Polygon):
        poly_a = Polygon(np.array(poly_a, dtype=np.float32).reshape([-1, 2]))
    if not isinstance(poly_b, Polygon):
        poly_b = Polygon(np.array(poly_b, dtype=np.float32).reshape([-1, 2]))

    if poly_a.is_valid and poly_b.is_valid:
        area_inters = poly_a.intersection(poly_b).area if poly_a.intersects(poly_b) else 0.0
        area_union = poly_a.union(poly_b).area

        iou = area_inters / area_union if area_union != 0 else zero_division
    else:
        iou = 0.0  # FIXME: calculate iou when poly_a or poly_b is invalid

    return iou
