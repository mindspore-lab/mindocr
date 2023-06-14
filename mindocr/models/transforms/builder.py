__all__ = ["build_trans"]
supported_trans = ["STN_ON"]

from .stn import STN_ON


def build_trans(trans_name, **kwargs):
    assert (
        trans_name in supported_trans
    ), f"Invalid transforms: {trans_name}, Support transforms are {supported_trans}"
    trans = eval(trans_name)(**kwargs)
    return trans
