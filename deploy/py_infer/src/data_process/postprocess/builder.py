from ..utils.process_parser import parse_from_yaml

__all__ = ["build_postprocess"]


class Postprocessor:
    def __init__(self, tasks, **kwargs):
        ops, params = list(tasks.items())[0]
        params.update(**kwargs)
        self._ops_func = ops(**params)

    def __call__(self, *args, **kwargs):
        return self._ops_func(*args, **kwargs)


def build_postprocess(config_path, **kwargs):
    _, tasks = parse_from_yaml(config_path)
    processor = Postprocessor(tasks, **kwargs)
    return processor
