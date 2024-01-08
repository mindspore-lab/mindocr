from ._registry import register_model
from .backbones.mindcv_models.utils import load_pretrained
from .base_model import BaseModel

__all__ = ['PSENet', 'psenet_resnet152', 'psenet_resnet50', 'psenet_mobilenetv3']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'input_size': (3, 640, 640),
        **kwargs
    }


default_cfgs = {
    'psenet_resnet152': _cfg(
        url='https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_resnet152_ic15-6058a798.ckpt'),
    'psenet_resnet50': _cfg(
            url='https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_resnet50_ic15-7e36cab9.ckpt'),
    'psenet_mobilenetv3': _cfg(
            url='https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_mobilenetv3_ic15-bf2c1907.ckpt'),
}


class PSENet(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def psenet_resnet152(pretrained=False, pretrained_backbone=True, **kwargs):
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


@register_model
def psenet_resnet50(pretrained=False, **kwargs):
    pretrained_backbone = not pretrained
    model_config = {
        "backbone": {
            'name': 'det_resnet50',
            'pretrained': pretrained_backbone
        },
        "neck": {
            "name": 'PSEFPN',
            "out_channels": 256,
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
        default_cfg = default_cfgs['psenet_resnet50']
        load_pretrained(model, default_cfg)

    return model


@register_model
def psenet_mobilenetv3(pretrained=False, **kwargs):
    pretrained_backbone = 'https://download.mindspore.cn/toolkits/mindcv/mobilenet/mobilenetv3' \
                          '/mobilenet_v3_large_050_no_scale_se_v2_expand-3c4047ac.ckpt'
    model_config = {
        "backbone": {
            'name': 'det_mobilenet_v3',
            'architecture': 'large',
            'alpha': 0.5,
            'out_stages': [5, 8, 14, 20],
            'bottleneck_params': {'se_version': 'SqueezeExciteV2', 'always_expand': True},
            'pretrained': pretrained_backbone if not pretrained else False
        },
        "neck": {
            "name": 'PSEFPN',
            "out_channels": 96,
        },
        "head": {
            "name": 'PSEHead',
            "hidden_size": 96,
            "out_channels": 7
        }
    }
    model = PSENet(model_config)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['psenet_mobilenetv3']
        load_pretrained(model, default_cfg)

    return model
