import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))

# from mindocr.infer.detection.det_pre_node import DetPreNode
# from mindocr.infer.detection.det_infer_node import DetInferNode
# from mindocr.infer.detection.det_post_node import DetPostNode
# from mindocr.infer.recognition.rec_pre_node import RecPreNode
# from mindocr.infer.recognition.rec_infer_node import RecInferNode
# from mindocr.infer.recognition.rec_post_node import RecPostNode
# from mindocr.infer.classification.cls_pre_node import ClsPreNode
# from mindocr.infer.classification.cls_infer_node import ClsInferNode
# from mindocr.infer.classification.cls_post_node import ClsPostNode
# from mindocr.infer.common.handout_node import HandoutNode
# from mindocr.infer.common.decode_node import DecodeNode
# from mindocr.infer.common.collect_node import CollectNode

from .detection import DetPreNode, DetInferNode, DetPostNode
from .recognition import RecPreNode, RecInferNode, RecPostNode
from .classification import ClsPreNode, ClsInferNode, ClsPostNode
from .common import HandoutNode, DecodeNode, CollectNode

from pipeline.tasks import TaskType
from pipeline.utils import log

__all__ = [
    "MODEL_DICT",
    "DET_DESC",
    "CLS_DESC",
    "REC_DESC",
    "DET_REC_DESC",
    "DET_CLS_REC_DESC",
]

DET_DESC = [
    (("HandoutNode", "0", 1), ("DecodeNode", "0", 1)),
    (("DecodeNode", "0", 1), ("DetPreNode", "0", 1)),
    (("DetPreNode", "0", 1), ("DetInferNode", "0", 1)),
    (("DetInferNode", "0", 1), ("DetPostNode", "0", 1)),
    (("DetPostNode", "0", 1), ("CollectNode", "0", 1)),
]

REC_DESC = [
    (("HandoutNode", "0", 1), ("DecodeNode", "0", 1)),
    (("DecodeNode", "0", 1), ("RecPreNode", "0", 1)),
    (("RecPreNode", "0", 1), ("RecInferNode", "0", 1)),
    (("RecInferNode", "0", 1), ("RecPostNode", "0", 1)),
    (("RecPostNode", "0", 1), ("CollectNode", "0", 1)),
]

CLS_DESC = [
    (("HandoutNode", "0", 1), ("DecodeNode", "0", 1)),
    (("DecodeNode", "0", 1), ("ClsPreNode", "0", 1)),
    (("ClsPreNode", "0", 1), ("ClsInferNode", "0", 1)),
    (("ClsInferNode", "0", 1), ("ClsPostNode", "0", 1)),
    (("ClsPostNode", "0", 1), ("CollectNode", "0", 1)),
]

DET_REC_DESC = [
    (("HandoutNode", "0", 1), ("DecodeNode", "0", 1)),
    (("DecodeNode", "0", 1), ("DetPreNode", "0", 1)),
    (("DetPreNode", "0", 1), ("DetInferNode", "0", 1)),
    (("DetInferNode", "0", 1), ("DetPostNode", "0", 1)),
    (("DetPostNode", "0", 1), ("RecPreNode", "0", 1)),
    (("RecPreNode", "0", 1), ("RecInferNode", "0", 1)),
    (("RecInferNode", "0", 1), ("RecPostNode", "0", 1)),
    (("RecPostNode", "0", 1), ("CollectNode", "0", 1)),
]

DET_CLS_REC_DESC = [
    (("HandoutNode", "0", 1), ("DecodeNode", "0", 1)),
    (("DecodeNode", "0", 1), ("DetPreNode", "0", 1)),
    (("DetPreNode", "0", 1), ("DetInferNode", "0", 1)),
    (("DetInferNode", "0", 1), ("DetPostNode", "0", 1)),
    (("DetPostNode", "0", 1), ("ClsPreNode", "0", 1)),
    (("ClsPreNode", "0", 1), ("ClsInferNode", "0", 1)),
    (("ClsInferNode", "0", 1), ("ClsPostNode", "0", 1)),
    (("ClsPostNode", "0", 1), ("RecPreNode", "0", 1)),
    (("RecPreNode", "0", 1), ("RecInferNode", "0", 1)),
    (("RecInferNode", "0", 1), ("RecPostNode", "0", 1)),
    (("RecPostNode", "0", 1), ("CollectNode", "0", 1)),
]

MODEL_DICT = {
    TaskType.DET: DET_DESC,
    TaskType.REC: REC_DESC,
    TaskType.CLS: CLS_DESC,
    TaskType.DET_REC: DET_REC_DESC,
    TaskType.DET_CLS_REC: DET_CLS_REC_DESC,
    # TaskType.LAYOUT: LAYOUT_DESC # TODO
}

def processor_initiator(classname):
    try:
        processor = getattr(sys.modules.get(__name__), classname)
    except AttributeError as error:
        log.error("%s doesn't exist.", classname)
        raise error
    if isinstance(processor, type):
        return processor
    raise TypeError("%s doesn't exist.", classname)
