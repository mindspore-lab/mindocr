from dataclasses import dataclass, field
from typing import List, Union, Dict

import numpy as np


@dataclass
class ProcessData:
    # infer_test info
    sub_image_total: int = 0
    image_total: int = 0
    infer_result: list = field(default_factory=lambda: [])
    skip: bool = False

    # image basic info
    image_path: str = ''
    image_name: str = ''
    image_id: int = ''
    frame: np.ndarray = None

    original_width: int = 0
    original_height: int = 0
    sub_image_list: list = field(default_factory=lambda: [])
    sub_image_size: int = 0
    data: Union[np.ndarray, List[np.ndarray], Dict] = None


@dataclass
class StopData:
    skip: bool = True
    image_total: int = 0
