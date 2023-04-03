from mindspore import nn
from .backbones.mindcv_models.utils import load_pretrained
from .base_model import BaseModel
from ._registry import register_model

__all__ = ['DBNet', 'dbnet_r50']

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'input_size': (3, 640, 640),
        **kwargs
    }


default_cfgs = {
    'dbnet_r50': _cfg(
        url='https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50-db1df47a.ckpt')
    }


class DBNet(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def dbnet_r50(pretrained=False, **kwargs):
    model_config = {
            "backbone": {
                'name': 'det_resnet50',
                'pretrained': False
                },
            "neck": {
                "name": 'DBFPN',
                "out_channels": 256,
                "bias": False,
                "use_asf": False        # enable it for DB++
                },
            "head": {
                "name": 'DBHead',
                "k": 50,
                "bias": False,
                "adaptive": True
                }
            }
    model = DBNet(model_config)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['dbnet_r50']
        load_pretrained(model, default_cfg)

    return model

'''
@register_model
def dbnet_mv3(pretrained=False, **kwargs):
    model_config = {
            "backbone": {
                'name': 'det_mv3',
                'pretrained': False
                },
            "neck": {
                "name": 'FPN',
                "out_channels": 128,
                },
            "head": {
                "name": 'ConvHead',
                "out_channels": 2,
                "k": 50
                }
            }
    model = DBNet(model_config)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['dbnet_mv3']
        load_pretrained(model, default_cfg)

    return model
'''
