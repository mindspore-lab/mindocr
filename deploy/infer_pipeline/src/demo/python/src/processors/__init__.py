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

from src.processors.classification.cls import CLSPreProcess, CLSInferProcess
from src.processors.common import HandoutProcess, CollectProcess, DecodeProcess
from src.processors.detection.dbnet import DBNetPreProcess, DBNetInferProcess, DBNetPostProcess
from src.processors.recognition.crnn import CRNNPreProcess, CRNNInferProcess, CRNNPostProcess
from src.utils import log

DBNET_DESC = [('DBNetPreProcess', 1), ('DBNetInferProcess', 1), ('DBNetPostProcess', 1)]
CRNN_DESC = [('CRNNPreProcess', 1), ('CRNNInferProcess', 1), ('CRNNPostProcess', 1)]
CLS_DESC = [('CLSPreProcess', 1), ('CLSInferProcess', 1)]

MODEL_DICT = {
    'dbnet': DBNET_DESC,
    'crnn': CRNN_DESC,
    'cls': CLS_DESC
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
