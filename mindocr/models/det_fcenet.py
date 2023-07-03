from ._registry import register_model
from .backbones.mindcv_models.utils import load_pretrained
from .base_model import BaseModel

__all__ = ['FCENet', 'fcenet_resnet50']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'input_size': (3, 640, 640),
        **kwargs
    }


default_cfgs = {
    'fcenet_resnet50': _cfg(
        url='https://download.mindspore.cn/toolkits/mindocr/fcenet/fcenet_resnet50-43857f7f.ckpt'),
}


class FCENet(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def fcenet_resnet50(pretrained=False, **kwargs):
    pretrained_backbone = not pretrained
    model_config = {
        "backbone": {
            'name': 'det_resnet50',
            'pretrained': pretrained_backbone
        },
        "neck": {
            "name": 'FCEFPN',
            "out_channels": 256,
        },
        "head": {
            "name": 'FCEHead',
            "scales": [8, 16, 32],
            "alpha": 1.2,
        }
    }
    model = FCENet(model_config)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['fcenet_resnet50']
        load_pretrained(model, default_cfg)

    return model
