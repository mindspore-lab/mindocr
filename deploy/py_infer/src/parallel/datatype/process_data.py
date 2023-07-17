from dataclasses import dataclass, field
from typing import Dict, List, Union

import numpy as np


@dataclass
class ProcessData:
    # skip each compute node
    skip: bool = False
    # prediction results of each image
    infer_result: list = field(default_factory=lambda: [])

    # image basic info
    image_path: List[str] = field(default_factory=lambda: [])
    frame: List[np.ndarray] = field(default_factory=lambda: [])

    # sub image of detection box, for det (+ cls) + rec
    sub_image_total: int = 0  # len(sub_image_list_0) + len(sub_image_list_1) + ...
    sub_image_list: list = field(default_factory=lambda: [])
    sub_image_size: int = 0  # len of sub_image_list

    # data for preprocess -> infer -> postprocess
    data: Union[np.ndarray, List[np.ndarray], Dict] = None


@dataclass
class StopData:
    skip: bool = True
    image_total: int = 0
    exception: bool = False
