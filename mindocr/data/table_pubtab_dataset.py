import json
import logging
import os
import random
from typing import Any, List, Optional

import numpy as np

from .base_dataset import BaseDataset
from .transforms.transforms_factory import create_transforms, run_transforms

_logger = logging.getLogger(__name__)


class PubTabDataset(BaseDataset):
    def __init__(
        self,
        is_train: bool = True,
        data_dir: str = "",
        label_file_list: str = "",
        sample_ratio_list: Optional[List[dict]] = [1.0],
        shuffle: Optional[bool] = None,
        transform_pipeline: Optional[List[dict]] = None,
        output_columns: Optional[List[str]] = None,
        max_text_len: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ):
        self.is_train = is_train
        self.data_dir = data_dir
        self.max_text_len = max_text_len

        self.do_shuffle = shuffle if shuffle is not None else is_train
        self.seed = seed

        self.data_list = self.get_image_info_list(label_file_list, sample_ratio_list)

        # create transform
        if transform_pipeline is not None:
            self.transforms = create_transforms(transform_pipeline)
        else:
            raise ValueError("No transform pipeline is specified!")
        self.prefetch(output_columns)

    def prefetch(self, output_columns):
        # prefetch the data keys, to fit GeneratorDataset
        _data_line = self.data_list[0]
        info = json.loads(_data_line)
        file_name = info["filename"]
        cells = info["html"]["cells"].copy()
        structure = info["html"]["structure"]["tokens"].copy()
        img_path = os.path.join(self.data_dir, file_name)
        _data = {"img_path": img_path, "cells": cells, "structure": structure, "file_name": file_name}
        _data = run_transforms(_data, transforms=self.transforms)
        _available_keys = list(_data.keys())

        if output_columns is None:
            self.output_columns = _available_keys
        else:
            self.output_columns = []
            for k in output_columns:
                if k in _data:
                    self.output_columns.append(k)
                else:
                    raise ValueError(
                        f"Key {k} does not exist in data (available keys: {_data.keys()}). "
                        "Please check the name or the completeness transformation pipeline."
                    )

    def get_image_info_list(self, file_list, sample_ratio_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_list = []
        for idx, file in enumerate(file_list):
            with open(file, "rb") as f:
                lines = f.readlines()
                if self.is_train is True or sample_ratio_list[idx] < 1.0:
                    random.seed(self.seed)
                    lines = random.sample(lines, round(len(lines) * sample_ratio_list[idx]))
                data_list.extend(lines)
        return data_list

    def shuffle_data_random(self):
        if self.do_shuffle:
            random.seed(self.seed)
            random.shuffle(self.data_list)
        return

    def __getitem__(self, idx):
        data_line = self.data_list[idx]
        info = json.loads(data_line)
        file_name = info["filename"]
        cells = info["html"]["cells"].copy()
        structure = info["html"]["structure"]["tokens"].copy()
        img_path = os.path.join(self.data_dir, file_name)
        if not os.path.exists(img_path):
            raise Exception("{} does not exist!".format(img_path))
        data = {"img_path": img_path, "cells": cells, "structure": structure, "file_name": file_name}
        try:
            outs = run_transforms(data, transforms=self.transforms)
        except RuntimeError:
            _logger.warning("data is None after transforms, random choose another data.")
            outs = None
        if outs is None:
            rnd_idx = np.random.randint(self.__len__()) if self.is_train is True else (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)

        output_tuple = tuple(outs[k] for k in self.output_columns)

        return output_tuple
