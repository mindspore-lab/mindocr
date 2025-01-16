from . import (  # rec_abinet_postprocess,
    cls_postprocess,
    det_db_postprocess,
    det_east_postprocess,
    det_pse_postprocess,
    kie_re_postprocess,
    kie_ser_postprocess,
    layout_postprocess,
    rec_abinet_postprocess,
    rec_postprocess,
    table_postprocess,
)
from .cls_postprocess import *
from .det_db_postprocess import *
from .det_east_postprocess import *
from .det_pse_postprocess import *
from .kie_re_postprocess import VQAReTokenLayoutLMPostProcess
from .kie_ser_postprocess import VQASerTokenLayoutLMPostProcess
from .layout_postprocess import YOLOv8Postprocess
from .rec_abinet_postprocess import *
from .rec_postprocess import *
from .table_postprocess import *

__all__ = ["build_postprocess"]

supported_postprocess = (
    det_db_postprocess.__all__
    + det_pse_postprocess.__all__
    + det_east_postprocess.__all__
    + rec_postprocess.__all__
    + cls_postprocess.__all__
    + rec_abinet_postprocess.__all__
    + kie_ser_postprocess.__all__
    + kie_re_postprocess.__all__
    + layout_postprocess.__all__
)


def build_postprocess(config: dict):
    """
    Create postprocess function.

    Args:
        config (dict): configuration for postprocess including postprocess `name` and also the kwargs specifically
        for each postprocessor.
            - name (str): metric function name, exactly the same as one of the supported postprocess class names

    Return:
        Object

    Example:
        >>> # Create postprocess function
        >>> from mindocr.postprocess import build_postprocess
        >>> config = dict(name="RecCTCLabelDecode", use_space_char=False)
        >>> postprocess = build_postprocess(config)
        >>> postprocess
    """
    proc = config.pop("name")
    if proc in supported_postprocess:
        postprocessor = eval(proc)(**config)
    elif proc is None:
        return None
    else:
        raise ValueError(f"Invalid postprocess name {proc}, support postprocess are {supported_postprocess}")

    return postprocessor
