from sys import modules

from ..framework import InferModelComb
from ..utils import log
from .classification import CLSInferProcess, CLSPreProcess
from .common import CollectProcess, DecodeProcess, HandoutProcess
from .detection import SUPPORT_DET_MODEL, DetInferProcess, DetPostProcess, DetPreProcess
from .recognition import SUPPORT_REC_MODEL, RecInferProcess, RecPostProcess, RecPreProcess

DET_DESC = [("DetPreProcess", 1), ("DetInferProcess", 1), ("DetPostProcess", 1)]
REC_DESC = [("RecPreProcess", 1), ("RecInferProcess", 1), ("RecPostProcess", 1)]
CLS_DESC = [("CLSPreProcess", 1), ("CLSInferProcess", 1)]

MODEL_DICT = {InferModelComb.DET: DET_DESC, InferModelComb.REC: REC_DESC, InferModelComb.CLS: CLS_DESC}


def processor_initiator(classname):
    try:
        processor = getattr(modules.get(__name__), classname)
    except AttributeError as error:
        log.error("%s doesn't exist.", classname)
        raise error
    if isinstance(processor, type):
        return processor
    raise TypeError("%s doesn't exist.", classname)
