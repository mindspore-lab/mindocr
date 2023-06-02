import yaml

from . import postprocess_mapping

__all__ = ["build_postprocess"]


class Postprocessor:
    def __init__(self, tasks, **kwargs):
        ops, params = list(tasks.items())[0]
        params.update(**kwargs)
        self._ops_func = ops(**params)

    def __call__(self, *args, **kwargs):
        return self._ops_func(*args, **kwargs)


def parse_postprocess_from_yaml(config_path):
    with open(config_path) as fp:
        cfg = yaml.safe_load(fp)

    ops_params = cfg['postprocess']

    ops_node = postprocess_mapping.POSTPROCESS_MAPPING_OPS[ops_params["name"]]
    ops_params.pop("name")
    ops = {ops_node: ops_params}

    return ops


def build_postprocess(config_path, **kwargs):
    tasks = parse_postprocess_from_yaml(config_path)
    processor = Postprocessor(tasks, **kwargs)
    return processor
