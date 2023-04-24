from .transforms import NormalizeImage, ToNCHW, ResizeKeepAspectRatio, LimitMaxSide, Resize, RGB2BGR
from ..utils import constant

__all__ = ["build_preprocess"]


class Preprocessor:
    def __init__(self, algorithm: str, ops_list: list, init_params: dict):
        self.algorithm = algorithm
        self._check_init_params(ops_list, init_params)
        self.ops_list = [ops(**init_params.get(ops.__name__, {})) for ops in ops_list]

    def _check_init_params(self, ops_list: list, init_params: dict):
        ops_names = {ops.__name__ for ops in ops_list}
        ops_params_names = set(init_params.keys())
        diff = ops_params_names - ops_names
        if diff:
            raise ValueError(
                f"Build preprocessor failed for {self.algorithm}, preprocessor is {ops_names}, "
                f"but parameter for preprocessor has {diff}.")

    def __call__(self, image, extra_params: dict = None):
        extra_params = {} if not extra_params else extra_params
        dst_image = image
        for ops in self.ops_list:
            ops_name = ops.__class__.__name__
            if ops_name in extra_params:
                dst_image = ops(dst_image, **extra_params[ops_name])
            else:
                dst_image = ops(dst_image)

        return dst_image


def build_preprocess(algorithm: str, init_params: dict = None):
    init_params = {} if not init_params else init_params

    ops_list = []
    algorithm = algorithm.lower()
    if algorithm in ("dbnet",):
        ops_list.append(Resize)
        params = {
            "NormalizeImage": {
                "std": constant.IMAGE_NET_IMAGE_STD,
                "mean": constant.IMAGE_NET_IMAGE_MEAN
            }
        }
        init_params.update(params)
    elif algorithm in ('cls',):
        ops_list.append(RGB2BGR)
        ops_list.append(Resize)
    elif algorithm in ("cls", "crnn", "svtr"):
        ops_list.append(Resize)
    else:
        raise ValueError(f"{algorithm} is not supported for preprocess.")

    ops_list.append(NormalizeImage)
    ops_list.append(ToNCHW)

    return Preprocessor(algorithm, ops_list, init_params)
