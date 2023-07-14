from ._registry import register_model
from .backbones.mindcv_models.utils import load_pretrained
from .base_model import BaseModel

__all__ = ['EAST', 'east_resnet50', 'east_mobilenetv3']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'input_size': (3, 640, 640),
        **kwargs
    }


default_cfgs = {
    'east_resnet50': _cfg(
        url='https://download.mindspore.cn/toolkits/mindocr/east/east_resnet50_ic15-7262e359.ckpt'),
    'east_mobilenetv3': _cfg(
        url='https://download.mindspore.cn/toolkits/mindocr/east/east_mobilenetv3_ic15-4288dba1.ckpt'),
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


@register_model
def east_mobilenetv3(pretrained=False, **kwargs):
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
        default_cfg = default_cfgs['east_mobilenetv3']
        load_pretrained(model, default_cfg)

    return model
