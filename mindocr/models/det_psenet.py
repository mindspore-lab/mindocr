from .base_model import BaseModel
from ._registry import register_model
from .backbones.mindcv_models.utils import load_pretrained


__all__ = ['PSENet', 'psenet_resnet152']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'input_size': (3, 640, 640),
        **kwargs
    }


default_cfgs = {
    'psenet_resnet152': _cfg(
        url='https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_resnet152_ic15-6058a798.ckpt')
}


class PSENet(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def psenet_resnet152(pretrained=False, **kwargs):
    pretrained_backbone = not pretrained
    model_config = {
        "backbone": {
            'name': 'det_resnet152',
            'pretrained': pretrained_backbone
        },
        "neck": {
            "name": 'PSEFPN',
            "out_channels": 128,
        },
        "head": {
            "name": 'PSEHead',
            "hidden_size": 256,
            "out_channels": 7
        }
    }
    model = PSENet(model_config)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['psenet_resnet152']
        load_pretrained(model, default_cfg)

    return model