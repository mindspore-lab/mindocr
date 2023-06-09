from ._registry import register_model
from .backbones.mindcv_models.utils import load_pretrained
from .base_model import BaseModel

__all__ = ['CRNN', 'crnn_resnet34', 'crnn_vgg7', 'crnn_resnet34_ch']


def _cfg(url='', input_size=(3, 32, 100), **kwargs):
    return {
        'url': url,
        'input_size': input_size,
        **kwargs
    }


default_cfgs = {
    'crnn_resnet34': _cfg(
        url='https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34-83f37f07.ckpt'),
    'crnn_vgg7': _cfg(
        url='https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_vgg7-ea7e996c.ckpt'),
    'crnn_resnet34_ch': _cfg(
        url='https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34_ch-7a342e3c.ckpt',
        input_size=(3, 32, 320)),
    }


class CRNN(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def crnn_resnet34(pretrained=False, **kwargs):
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
        default_cfg = default_cfgs['crnn_resnet34']
        load_pretrained(model, default_cfg)

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
        default_cfg = default_cfgs['crnn_vgg7']
        load_pretrained(model, default_cfg)

    return model


@register_model
def crnn_resnet34_ch(pretrained=False, **kwargs):
    model_config = {
        "backbone": {
            'name': 'rec_resnet34',
            'pretrained': False
        },
        "neck": {
            "name": 'RNNEncoder',
            "hidden_size": 64,
        },
        "head": {
            "name": 'CTCHead',
            "out_channels": 6624,
            "weight_init": "crnn_customised",
            "bias_init": "crnn_customised",
        }
    }
    model = CRNN(model_config)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['crnn_resnet34_ch']
        load_pretrained(model, default_cfg)

    return model
