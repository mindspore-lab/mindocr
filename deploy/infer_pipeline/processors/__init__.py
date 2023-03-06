#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
Description:
Author: MindX SDK
Create: 2022
History: NA
"""

from sys import modules

from deploy.infer_pipeline.utils import log
from deploy.infer_pipeline.framework import InferModelComb

from .classification import CLSPreProcess, CLSInferProcess
from .common import HandoutProcess, CollectProcess, DecodeProcess
from .detection import DetPreProcess, DetInferProcess, DetPostProcess, SUPPORT_DET_MODEL
from .recognition import RecPreProcess, RecInferProcess, RecPostProcess, SUPPORT_REC_MODEL


DET_DESC = [('DetPreProcess', 1), ('DetInferProcess', 1), ('DetPostProcess', 1)]
REC_DESC = [('RecPreProcess', 1), ('RecInferProcess', 1), ('RecPostProcess', 1)]
CLS_DESC = [('CLSPreProcess', 1), ('CLSInferProcess', 1)]

MODEL_DICT = {
    InferModelComb.DET: DET_DESC,
    InferModelComb.REC: REC_DESC,
    InferModelComb.CLS: CLS_DESC
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
