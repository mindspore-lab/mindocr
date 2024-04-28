from sys import modules

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))

from mindocr.infer.detection import DetInferNode, DetPostNode, DetPreNode
from mindocr.infer.recognition import RecInferNode, RecPostNode, RecPreNode
from mindocr.infer.classification import ClsPreNode, ClsInferNode, ClsPostNode
from mindocr.infer.layout import LayoutPreNode, LayoutInferNode, LayoutPostNode, LayoutCollectNode
from mindocr.infer.common import CollectNode, DecodeNode, HandoutNode
# from deploy.py_infer.src.infer import TaskType
from pipeline.tasks import TaskType
from pipeline.utils import log

__all__ = ["MODEL_DICT_v2",
           "DET_DESC_v2", "CLS_DESC_v2", "REC_DESC_v2",
           "DET_REC_DESC_v2", "DET_CLS_REC_DESC_v2",
           "LAYOUT_DESC_v2", "LAYOUT_DET_REC_DESC_v2", "LAYOUT_DET_CLS_REC_DESC_v2"]

DET_DESC_v2 = [
    (("HandoutNode", "0", 1), ("DecodeNode", "0", 1)),
    (("DecodeNode", "0", 1), ("DetPreNode", "0", 1)),
    (("DetPreNode", "0", 1), ("DetInferNode", "0", 1)),
    (("DetInferNode", "0", 1), ("DetPostNode", "0", 1)),
    (("DetPostNode", "0", 1), ("CollectNode", "0", 1)),
]

REC_DESC_v2 = [
    (("HandoutNode", "0", 1), ("DecodeNode", "0", 1)),
    (("DecodeNode", "0", 1), ("RecPreNode", "0", 1)),
    (("RecPreNode", "0", 1), ("RecInferNode", "0", 1)),
    (("RecInferNode", "0", 1), ("RecPostNode", "0", 1)),
    (("RecPostNode", "0", 1), ("CollectNode", "0", 1)),
]

CLS_DESC_v2 = [
    (("HandoutNode", "0", 1), ("DecodeNode", "0", 1)),
    (("DecodeNode", "0", 1), ("ClsPreNode", "0", 1)),
    (("ClsPreNode", "0", 1), ("ClsInferNode", "0", 1)),
    (("ClsInferNode", "0", 1), ("ClsPostNode", "0", 1)),
    (("ClsPostNode", "0", 1), ("CollectNode", "0", 1)),
]

DET_REC_DESC_v2 = [
    (("HandoutNode", "0", 1), ("DecodeNode", "0", 1)),
    (("DecodeNode", "0", 1), ("DetPreNode", "0", 1)),
    (("DetPreNode", "0", 1), ("DetInferNode", "0", 1)),
    (("DetInferNode", "0", 1), ("DetPostNode", "0", 1)),
    (("DetPostNode", "0", 1), ("RecPreNode", "0", 1)),
    (("RecPreNode", "0", 1), ("RecInferNode", "0", 1)),
    (("RecInferNode", "0", 1), ("RecPostNode", "0", 1)),
    (("RecPostNode", "0", 1), ("CollectNode", "0", 1)),
]

DET_CLS_REC_DESC_v2 = [
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

LAYOUT_DESC_v2 = [
    (("HandoutNode", "0", 1), ("DecodeNode", "0", 1)),
    (("DecodeNode", "0", 1), ("LayoutPreNode", "0", 1)),
    (("LayoutPreNode", "0", 1), ("LayoutInferNode", "0", 1)),
    (("LayoutInferNode", "0", 1), ("LayoutPostNode", "0", 1)),
    (("LayoutPostNode", "0", 1), ("CollectNode", "0", 1)),
]

LAYOUT_DET_REC_DESC_v2 = [
    (("HandoutNode", "0", 1), ("DecodeNode", "0", 1)),
    (("DecodeNode", "0", 1), ("LayoutPreNode", "0", 1)),
    (("LayoutPreNode", "0", 1), ("LayoutInferNode", "0", 1)),
    (("LayoutInferNode", "0", 1), ("LayoutPostNode", "0", 1)),
    (("LayoutPostNode", "0", 1), ("DetPreNode", "0", 1)),
    (("DetPreNode", "0", 1), ("DetInferNode", "0", 1)),
    (("DetInferNode", "0", 1), ("DetPostNode", "0", 1)),
    (("DetPostNode", "0", 1), ("RecPreNode", "0", 1)),
    (("RecPreNode", "0", 1), ("RecInferNode", "0", 1)),
    (("RecInferNode", "0", 1), ("RecPostNode", "0", 1)),
    (("RecPostNode", "0", 1), ("LayoutCollectNode", "0", 1)),
    (("LayoutCollectNode", "0", 1), ("CollectNode", "0", 1)),
]

LAYOUT_DET_CLS_REC_DESC_v2 = [
    (("HandoutNode", "0", 1), ("DecodeNode", "0", 1)),
    (("DecodeNode", "0", 1), ("LayoutPreNode", "0", 1)),
    (("LayoutPreNode", "0", 1), ("LayoutInferNode", "0", 1)),
    (("LayoutInferNode", "0", 1), ("LayoutPostNode", "0", 1)),
    (("LayoutPostNode", "0", 1), ("DetPreNode", "0", 1)),
    (("DetPreNode", "0", 1), ("DetInferNode", "0", 1)),
    (("DetInferNode", "0", 1), ("DetPostNode", "0", 1)),
    (("DetPostNode", "0", 1), ("ClsPreNode", "0", 1)),
    (("ClsPreNode", "0", 1), ("ClsInferNode", "0", 1)),
    (("ClsInferNode", "0", 1), ("ClsPostNode", "0", 1)),
    (("ClsPostNode", "0", 1), ("RecPreNode", "0", 1)),
    (("RecPreNode", "0", 1), ("RecInferNode", "0", 1)),
    (("RecInferNode", "0", 1), ("RecPostNode", "0", 1)),
    (("RecPostNode", "0", 1), ("LayoutCollectNode", "0", 1)),
    (("LayoutCollectNode", "0", 1), ("CollectNode", "0", 1)),
]

MODEL_DICT_v2 = {TaskType.DET: DET_DESC_v2,
                 TaskType.CLS: CLS_DESC_v2,
                 TaskType.REC: REC_DESC_v2,
                 TaskType.DET_REC: DET_REC_DESC_v2,
                 TaskType.DET_CLS_REC: DET_CLS_REC_DESC_v2,
                 TaskType.LAYOUT: LAYOUT_DESC_v2,
                 TaskType.LAYOUT_DET_REC: LAYOUT_DET_REC_DESC_v2,
                 TaskType.LAYOUT_DET_CLS_REC: LAYOUT_DET_CLS_REC_DESC_v2,}

def processor_initiator(classname):
    try:
        processor = getattr(modules.get(__name__), classname)
    except AttributeError as error:
        log.error("%s doesn't exist.", classname)
        raise error
    if isinstance(processor, type):
        return processor
    raise TypeError("%s doesn't exist.", classname)