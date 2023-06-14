from ._registry import register_model
from .backbones.mindcv_models.utils import load_pretrained
from .base_model import BaseModel

__all__ = ['ClsMv3', 'cls_mobilenet_v3_small_100_model']


def _cfg(url='', input_size=(3, 48, 192), **kwargs):
    return {
        'url': url,
        'input_size': input_size,
        **kwargs
    }


default_cfgs = {
    "mobilenet_v3_small_1.0": _cfg(
        url="https://download.mindspore.cn/toolkits/mindocr/cls/cls_mobilenetv3-92db9c58.ckpt"),
    # "mobilenet_v3_large_1.0": _cfg(
    #     url="")
}


class ClsMv3(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def cls_mobilenet_v3_small_100_model(pretrained=False, **kwargs):
    pretrained_backbone = not pretrained
    model_config = {
        "backbone": {
            'name': 'cls_mobilenet_v3_small_100',
            'pretrained': pretrained_backbone,  # backbone pretrained
        },
        "head": {
            "name": 'ClsHead',
            "hidden_channels": 1024,
            "num_classes": 2,
        }
    }
    model = ClsMv3(model_config)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['mobilenet_v3_small_1.0']
        load_pretrained(model, default_cfg)

    return model
