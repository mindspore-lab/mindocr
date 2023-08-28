from ._registry import register_model
from .backbones.mindcv_models.utils import load_pretrained
from .base_model import BaseModel

__all__ = ['VISIONLAN', 'visionlan_resnet45']


def _cfg(url='', input_size=(3, 32, 100), **kwargs):
    return {
        'url': url,
        'input_size': input_size,
        **kwargs
    }


default_cfgs = {
    "visionlan_resnet45": _cfg(
        url="https://download.mindspore.cn/toolkits/mindocr/visionlan/visionlan_resnet45_LA-e9720d9e.ckpt",
        input_size=(3, 64, 256),
    )
    }


class VISIONLAN(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def visionlan_resnet45(pretrained=False, **kwargs):
    model_config = {
        "backbone": {
            'name': 'rec_resnet45',
            'pretrained': False,
            'strides': [2, 2, 2, 1, 1]
        },
        "head": {
            'name': 'VisionLANHead',
            'n_layers': 3,
            'n_position': 256,
            'n_dim': 512,
            'max_text_length':  25,
            'training_step': "LA"
        }
    }
    model = VISIONLAN(model_config)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['visionlan_resnet45']
        load_pretrained(model, default_cfg)

    return model
