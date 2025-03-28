from . import _registry, builder
from ._registry import *
from .builder import *
from .cls_mv3 import *
from .det_dbnet import *
from .det_east import *
from .det_psenet import *
from .kie_layoutxlm import *
from .layout_yolov8 import *
from .rec_abinet import *
from .rec_crnn import *
from .rec_master import *
from .rec_rare import *
from .rec_robustscanner import *
from .rec_svtr import *
from .rec_visionlan import *
from .table_master import *

__all__ = []
__all__.extend(builder.__all__)
__all__.extend(_registry.__all__)
