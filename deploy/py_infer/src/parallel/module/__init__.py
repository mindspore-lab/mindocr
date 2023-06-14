from sys import modules

from ...infer import TaskType
from ...utils import log
from .classification import ClsInferNode, ClsPreNode
from .common import CollectNode, DecodeNode, HandoutNode
from .detection import DetInferNode, DetPostNode, DetPreNode
from .recognition import RecInferNode, RecPostNode, RecPreNode

DET_DESC = [("DetPreNode", 1), ("DetInferNode", 1), ("DetPostNode", 1)]
REC_DESC = [("RecPreNode", 1), ("RecInferNode", 1), ("RecPostNode", 1)]
CLS_DESC = [("ClsPreNode", 1), ("ClsInferNode", 1)]

MODEL_DICT = {TaskType.DET: DET_DESC, TaskType.REC: REC_DESC, TaskType.CLS: CLS_DESC}


def processor_initiator(classname):
    try:
        processor = getattr(modules.get(__name__), classname)
    except AttributeError as error:
        log.error("%s doesn't exist.", classname)
        raise error
    if isinstance(processor, type):
        return processor
    raise TypeError("%s doesn't exist.", classname)
