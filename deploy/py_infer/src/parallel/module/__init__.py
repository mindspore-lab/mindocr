from sys import modules

from .classification import ClsPreNode, ClsInferNode
from .common import HandoutNode, CollectNode, DecodeNode
from .detection import DetPreNode, DetInferNode, DetPostNode
from .recognition import RecPreNode, RecInferNode, RecPostNode
from ...infer import TaskType
from ...utils import log

DET_DESC = [('DetPreNode', 1), ('DetInferNode', 1), ('DetPostNode', 1)]
REC_DESC = [('RecPreNode', 1), ('RecInferNode', 1), ('RecPostNode', 1)]
CLS_DESC = [('ClsPreNode', 1), ('ClsInferNode', 1)]

MODEL_DICT = {
    TaskType.DET: DET_DESC,
    TaskType.REC: REC_DESC,
    TaskType.CLS: CLS_DESC
}


def processor_initiator(classname):
    try:
        processor = getattr(modules.get(__name__), classname)
    except AttributeError as error:
        log.error("%s doesn't exist.", classname)
        raise error
    if isinstance(processor, type):
        return processor
    raise TypeError("%s doesn't exist.", classname)
