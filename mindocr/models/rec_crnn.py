import mindspore.nn as nn
from mindspore.ops import operations as ops
from .base_model import BaseModel
from ._registry import register_model


class CRNN(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def crnn_r34(pretrained=False, **kwargs):
    model_config = {
        "backbone": {
            'name': 'rec_resnet34',
            'pretrained': False
        },
        "neck": {
            "name": 'RNNEncoder',
            "hidden_size": 256,
        },
        "head": {
            "name": 'CTCHead',
            "out_channels": 37,
            "weight_init": "crnn_customised",
            "bias_init": "crnn_customised",
        }
    }
    model = CRNN(model_config)

    # load pretrained weights
    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def crnn_vgg7(pretrained=False, **kwargs):
    model_config = {
        "backbone": {
            'name': 'rec_vgg7',
            'pretrained': False
        },
        "neck": {
            "name": 'RNNEncoder',
            "hidden_size": 256,
        },
        "head": {
            "name": 'CTCHead',
            "out_channels": 37,
            "weight_init": "crnn_customised",
            "bias_init": "crnn_customised",
        }
    }
    model = CRNN(model_config)

    # load pretrained weights
    if pretrained:
        raise NotImplementedError()

    return model
