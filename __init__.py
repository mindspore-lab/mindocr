from . import mindocr

from .mindocr import *
from .mindocr import data, losses, metrics, models, postprocess, utils

__all__ = []
__all__.extend(mindocr.__all__)
