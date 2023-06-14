from .db_postprocess import DBPostProcess
from .rec_postprocess import RecCTCLabelDecode

__all__ = ["build_postprocess"]


class Postprocessor:
    def __init__(self, algorithm: str, ops: type, init_params: dict):
        self.ops = ops(**init_params)

    def __call__(self, *args, **kwargs):
        return self.ops(*args, **kwargs)


def build_postprocess(algorithm, init_params: dict = None):
    init_params = {} if not init_params else init_params

    algorithm = algorithm.lower()
    if algorithm in ("dbnet",):
        ops = DBPostProcess
    elif algorithm in ("crnn", "svtr"):
        ops = RecCTCLabelDecode
    else:
        raise ValueError(f"{algorithm} is not supported for postprocess.")

    return Postprocessor(algorithm, ops, init_params)
