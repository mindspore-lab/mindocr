from .base_model import BaseModel
from ._registry import register_model
from .backbones.mindcv_models.utils import load_pretrained


__all__ = ['RARE', 'rare_resnet34']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'input_size': (3, 32, 100),
        **kwargs
    }


default_cfgs = {
    'rare_resnet34': _cfg(
        url='https://download.mindspore.cn/toolkits/mindocr/rare/rare_resnet34-309dc63e.ckpt')
}


class RARE(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def rare_resnet34(pretrained=False, **kwargs):
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
            "name": 'AttentionHead',
            "hidden_size": 256,
            "out_channels": 38,
        }
    }
    model = RARE(model_config)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['rare_resnet34']
        load_pretrained(model, default_cfg)

    return model
