from ._registry import register_model
from .backbones.mindcv_models.utils import load_pretrained
from .base_model import BaseModel

__all__ = ["ABINetModel", "abinet"]


def _cfg(url="", **kwargs):
    return {"url": url, "input_size": (3, 32, 100), **kwargs}


default_cfgs = {
    # 'abinet':
    "abinet": _cfg(
        url="https://download-mindspore.osinfra.cn/toolkits/mindocr/abinet/abinet_resnet45_en-7efa1184.ckpt"
    ),
}


class ABINetModel(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def abinet(pretrained=False, **kwargs):
    model_config = {
        "backbone": {"name": "abinet_backbone", "pretrained": False},
        "head": {
            "name": "ABINetHead",
        },
    }
    model = ABINetModel(model_config)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['abinet']
        load_pretrained(model, default_cfg)

    return model
