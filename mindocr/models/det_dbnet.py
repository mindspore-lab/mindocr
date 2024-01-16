from ._registry import register_model
from .backbones.mindcv_models.utils import load_pretrained
from .base_model import BaseModel

__all__ = ['DBNet', 'dbnet_resnet50', 'dbnet_resnet18', 'dbnet_ppocrv3']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'input_size': (3, 640, 640),
        **kwargs
    }


default_cfgs = {
    'dbnet_mobilenetv3': _cfg(
        url='https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_mobilenetv3-62c44539.ckpt'),
    'dbnet_resnet18': _cfg(
        url='https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18-0c0c4cfa.ckpt'),
    'dbnet_resnet50': _cfg(
        url='https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50-c3a4aa24.ckpt'),
    'dbnet_resnet50_ch_en': _cfg(
        url='https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_ch_en_general-a5dbb141.ckpt'),
    'dbnetpp_resnet50': _cfg(
        url='https://download-mindspore.osinfra.cn/toolkits/mindocr/dbnet/dbnetpp_resnet50-068166c2.ckpt'),
    'dbnetpp_resnet50_ch_en': _cfg(
        url='https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnetpp_resnet50_ch_en_general-884ba5b9.ckpt'),
    'dbnet_ppocrv3': _cfg(
        url='https://download-mindspore.osinfra.cn/toolkits/mindocr/dbnet/dbnet_mobilenetv3_ppocrv3-70d6018f.ckpt')
}


class DBNet(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def dbnet_mobilenetv3(pretrained=False, pretrained_backbone=True, **kwargs):
    backbone_ckpt_url = 'https://download.mindspore.cn/toolkits/mindcv/mobilenet/mobilenetv3' \
                          '/mobilenet_v3_large_050_no_scale_se_v2_expand-3c4047ac.ckpt'
    model_config = {
        "backbone": {
            'name': 'det_mobilenet_v3',
            'architecture': 'large',
            'alpha': 0.5,
            'out_stages': [5, 8, 14, 20],
            'bottleneck_params': {'se_version': 'SqueezeExciteV2', 'always_expand': True},
            'pretrained': backbone_ckpt_url if pretrained_backbone else False
        },
        "neck": {
            "name": 'DBFPN',
            "out_channels": 256,
            "bias": False,
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
        default_cfg = default_cfgs['dbnet_mobilenetv3']
        load_pretrained(model, default_cfg)

    return model


@register_model
def dbnet_resnet18(pretrained=False, pretrained_backbone=True, **kwargs):
    model_config = {
        "backbone": {
            'name': 'det_resnet18',
            'pretrained': pretrained_backbone
        },
        "neck": {
            "name": 'DBFPN',
            "out_channels": 256,
            "bias": False,
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
        default_cfg = default_cfgs['dbnet_resnet18']
        load_pretrained(model, default_cfg)

    return model


@register_model
def dbnet_resnet50(pretrained=False, pretrained_backbone=True, **kwargs):
    model_config = {
        "backbone": {
            'name': 'det_resnet50',
            'pretrained': pretrained_backbone
        },
        "neck": {
            "name": 'DBFPN',
            "out_channels": 256,
            "bias": False,
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
        default_cfg = default_cfgs['dbnet_resnet50']
        load_pretrained(model, default_cfg)

    return model


@register_model
def dbnet_resnet50_ch_en(pretrained=False, pretrained_backbone=True, **kwargs):
    model_config = {
        "backbone": {
            'name': 'det_resnet50',
            'pretrained': pretrained_backbone
        },
        "neck": {
            "name": 'DBFPN',
            "out_channels": 256,
            "bias": False,
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
        default_cfg = default_cfgs['dbnet_resnet50_ch_en']
        load_pretrained(model, default_cfg)

    return model


@register_model
def dbnetpp_resnet50(pretrained=False, pretrained_backbone=True, **kwargs):
    model_config = {
        "backbone": {
            'name': 'det_resnet50',
            'pretrained': pretrained_backbone
        },
        "neck": {
            "name": 'DBFPN',
            "out_channels": 256,
            "bias": False,
            "use_asf": True,
            "channel_attention": True
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
        default_cfg = default_cfgs['dbnetpp_resnet50']
        load_pretrained(model, default_cfg)

    return model


@register_model
def dbnetpp_resnet50_ch_en(pretrained=False, pretrained_backbone=True, **kwargs):
    model_config = {
        "backbone": {
            'name': 'det_resnet50',
            'pretrained': pretrained_backbone
        },
        "neck": {
            "name": 'DBFPN',
            "out_channels": 256,
            "bias": False,
            "use_asf": True,
            "channel_attention": True
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
        default_cfg = default_cfgs['dbnetpp_resnet50_ch_en']
        load_pretrained(model, default_cfg)

    return model


@register_model
def dbnet_ppocrv3(pretrained=False, **kwargs):
    model_config = {
        "backbone": {
            'name': 'det_mobilenet_v3_enhance',
            'architecture': 'large',
            'alpha': 0.5,
            'disable_se': True,
            'pretrained': False
        },
        "neck": {
            "name": 'RSEFPN',
            "out_channels": 96,
            "shortcut": True,
        },
        "head": {
            "name": 'DBHeadEnhance',
            "k": 50,
            "bias": False,
            "adaptive": True
        }
    }
    model = DBNet(model_config)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['dbnet_ppocrv3']
        load_pretrained(model, default_cfg)

    return model
