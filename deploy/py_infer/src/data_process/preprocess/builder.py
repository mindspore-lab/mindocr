from .transforms import ToBatch
from ..utils.process_parser import parse_from_yaml

__all__ = ["build_preprocess"]


class Preprocessor:
    def __init__(self, tasks):
        self._ops_list = []
        for ops, params in tasks.items():
            self._ops_list.append(ops(**params))

    def _ops_func(self, data):
        for ops in self._ops_list:
            data = ops(data)
        return data

    def __call__(self, img_list, **kwargs):
        img_list = [img_list] if not isinstance(img_list, (tuple, list)) else img_list
        data_list = [{"image": img, **kwargs} for img in img_list]
        outputs = [self._ops_func(data) for data in data_list]

        return ToBatch()(outputs)


def build_preprocess(config_path):
    tasks, _ = parse_from_yaml(config_path)
    processor = Preprocessor(tasks)
    return processor
