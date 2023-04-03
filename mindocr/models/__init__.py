from . import builder, _registry

from .builder import * 
from ._registry import *

from .det_dbnet import *
from .rec_crnn import *

__all__ = []
__all__.extend(builder.__all__)
__all__.extend(_registry.__all__)