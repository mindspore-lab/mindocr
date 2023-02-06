from mindspore import nn
from mindcv.models.utils import load_pretrained
from .base_model import BaseModel
from ._registry import register_model

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'input_size': (3, 640, 640), 
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        **kwargs
    }


default_cfgs = {
    # ResNet and Wide ResNet
    'dbnet_r50': _cfg(
        url='https://download.mindspore.cn/toolkits/mindocr/det/dbnet_r50-133e1234.ckpt')
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
                "name": 'FPN',
                "out_channels": 256,
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
