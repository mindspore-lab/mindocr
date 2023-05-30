from collections import OrderedDict

import yaml

from . import preprocess_mapping
from .transforms import ToBatch

__all__ = ["build_preprocess"]


class Preprocessor:
    def __init__(self, tasks):
        self.output_columns = {"image", "shape_list"}
        self._ops_list = []
        for ops, params in tasks.items():
            self._ops_list.append(ops(**params))

    def _ops_func(self, data):
        for ops in self._ops_list:
            data = ops(data)
        return data

    def __call__(self, img_list, **kwargs):
        img_list = [img_list] if not isinstance(img_list, (tuple, list)) else img_list
        data_list = [{"image": img, "raw_img_shape": img.shape[:2], **kwargs} for img in img_list]
        outputs = [self._ops_func(data) for data in data_list]

        return ToBatch(self.output_columns)(outputs)


def parse_preprocess_from_yaml(config_path):
    with open(config_path) as fp:
        cfg = yaml.safe_load(fp)

    src_ops_pipeline = cfg['eval']['dataset']['transform_pipeline']
    tgt_ops_pipeline = OrderedDict()

    for src_ops in src_ops_pipeline:
        src_ops_name, src_ops_params = list(src_ops.items())[0]

        # Skip nodes outside of MAPPING_OPS
        if src_ops_name not in preprocess_mapping.PREPROCESS_MAPPING_OPS:
            continue

        tgt_ops = preprocess_mapping.PREPROCESS_MAPPING_OPS[src_ops_name]

        tgt_ops_params = src_ops_params if src_ops_params else {}
        tgt_ops_pipeline[tgt_ops] = tgt_ops_params

    return tgt_ops_pipeline


def build_preprocess(config_path):
    tasks = parse_preprocess_from_yaml(config_path)
    processor = Preprocessor(tasks)
    return processor
