import os
import sys
from collections import OrderedDict, defaultdict
from typing import List

import numpy as np
import yaml

from . import adapted_preprocess

mindocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
sys.path.insert(0, mindocr_path)

from mindocr.data.transforms import transforms_factory as mindocr_preprocess  # noqa

__all__ = ["build_preprocess"]


class Preprocessor:
    def __init__(self, tasks, input_columns):
        self._input_columns = input_columns
        self._other_columns = ["shape_list"]  # for det

        self._ops_list = []
        for ops, params in tasks.items():
            self._ops_list.append(ops(**params))

        self._to_batch_columns = self._input_columns + self._other_columns

    def __repr__(self):
        info = "Data Preprocess Pipeline: "
        info += " -> ".join([node.__class__.__name__ for node in self._ops_list])
        return info

    def _ops_func(self, data):
        for ops in self._ops_list:
            data = ops(data)
        return data

    def _to_batch(self, data_list: List[dict]) -> dict:
        batch_data = defaultdict(list)
        for data in data_list:
            for key, value in data.items():
                if key in self._to_batch_columns:
                    batch_data[key].append(value)

        outputs = {}
        for key, value in batch_data.items():
            outputs[key] = np.array(value)

        return outputs

    def __call__(self, img_list: List[np.ndarray], **kwargs):
        data_list = [{"image": img, "raw_img_shape": img.shape[:2], **kwargs} for img in img_list]
        outputs = [self._ops_func(data) for data in data_list]
        batch_data = self._to_batch(outputs)

        net_inputs = [batch_data[name] for name in self._input_columns]
        other_params = {name: batch_data[name] for name in self._other_columns if name in batch_data}

        return {"net_inputs": net_inputs, **other_params}


def parse_preprocess_from_yaml(config_path, support_gear=False):
    with open(config_path) as fp:
        cfg = yaml.safe_load(fp)

    dataset_cfg: dict = cfg["eval"]["dataset"]
    transform_pipeline = dataset_cfg["transform_pipeline"]
    infer_transform_pipeline = OrderedDict()

    for node in transform_pipeline:
        node_name, node_params = list(node.items())[0]

        # Skip nodes with 'Label' in name
        if "Label" in node_name:
            continue

        load_adapted_ops = support_gear or (
            not support_gear and node_name not in adapted_preprocess.gear_supported_list
        )

        # adapted_preprocess > mindocr_preprocess
        if hasattr(adapted_preprocess, node_name) and load_adapted_ops:
            node_instance = getattr(adapted_preprocess, node_name)
        elif hasattr(mindocr_preprocess, node_name):
            node_instance = getattr(mindocr_preprocess, node_name)
        else:
            raise ValueError(f"The preprocess '{node_name}' is not supported yet.")

        node_cls_params = node_params if node_params else {}
        infer_transform_pipeline[node_instance] = node_cls_params

    if "output_columns" in dataset_cfg and "net_input_column_index" in dataset_cfg:
        output_columns = dataset_cfg["output_columns"]
        columns_index = dataset_cfg["net_input_column_index"]
        input_columns = [output_columns[i] for i in columns_index]
    else:
        input_columns = ["image"]

    return infer_transform_pipeline, input_columns


def build_preprocess(config_path, support_gear=False):
    tasks, input_columns = parse_preprocess_from_yaml(config_path, support_gear)
    processor = Preprocessor(tasks, input_columns)
    return processor
