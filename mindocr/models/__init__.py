from . import _registry, builder
from ._registry import *
from .builder import *
from .cls_mv3 import *
from .det_dbnet import *
from .det_east import *
from .det_psenet import *
from .rec_crnn import *
from .rec_master import *
from .rec_rare import *
from .rec_svtr import *
from .rec_visionlan import *

__all__ = []
__all__.extend(builder.__all__)
__all__.extend(_registry.__all__)
