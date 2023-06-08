from ._registry import register_model
from .backbones.mindcv_models.utils import load_pretrained
from .base_model import BaseModel

__all__ = ['EAST', 'east_resnet50']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'input_size': (3, 640, 640),
        **kwargs
    }


default_cfgs = {
    'east_resnet50': _cfg(
        url='https://download.mindspore.cn/toolkits/mindocr/east/east_resnet50_ic15-7262e359.ckpt'),
}


class EAST(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def east_resnet50(pretrained=False, **kwargs):
    pretrained_backbone = not pretrained
    model_config = {
        "backbone": {
            'name': 'det_resnet50',
            'pretrained': pretrained_backbone
        },
        "neck": {
            "name": 'EASTFPN',
            "out_channels": 128
        },
        "head": {
            'name': 'EASTHead'
        }
    }
    model = EAST(model_config)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['east_resnet50']
        load_pretrained(model, default_cfg)

    return model
