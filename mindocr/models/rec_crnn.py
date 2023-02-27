import mindspore.nn as nn
from mindspore.ops import operations as ops
from .base_model import BaseModel
from ._registry import register_model


class CRNN(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def construct(self, x):
        x = self.backbone(x)
        bs, fs, _, _ = x.shape
        x = self.reshape(x, (bs, fs, -1))
        x = self.transpose(x, (0, 2, 1))
        x = self.neck(x)
        x = self.head(x)
        return x


@register_model
def crnn_r34(pretrained=False, **kwargs):
    model_config = {
        "backbone": {
            'name': 'rec_resnet34',
            'pretrained': False
        },
        "neck": {
            "name": 'RNNEncoder',
            "out_channels": 512,
            "hidden_size": 256,
            "batch_size": 8,
            "num_layers": 2,
            "dropout": 0.0,
            "bidirectional": True,
        },
        "head": {
            "name": 'CTCHead',
            "out_channels": 37,
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
            "out_channels": 512,
            "hidden_size": 256,
            "batch_size": 8,
            "num_layers": 2,
            "dropout": 0.0,
            "bidirectional": True,
        },
        "head": {
            "name": 'CTCHead',
            "out_channels": 37,
        }
    }
    model = CRNN(model_config)

    # load pretrained weights
    if pretrained:
        raise NotImplementedError()

    return model
