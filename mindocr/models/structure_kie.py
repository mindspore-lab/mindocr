import os

from ._registry import register_model
from .base_model import BaseModel

__all__ = ["KieNet", "layoutxlm_re", "layoutxlm_ser"]


def _cfg(url="", input_size=(3, 224, 224), **kwargs):
    return {
        "url": url,
        "input_size": input_size,
        **kwargs
    }


default_cfgs = {
    "layoutxlm_base": _cfg(
        url="https://download.mindspore.cn/toolkits/mindocr/cls/cls_mobilenetv3-92db9c58.ckpt")
}


class KieNet(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def layoutxlm_re(pretrained: bool = True, **kwargs) -> KieNet:
    os.environ["MS_DEV_JIT_SYNTAX_LEVEL"] = "0"
    pretrained_backbone = not pretrained
    model_config = {
        "type": "kie",
        "backbone": {
            "name": "layoutxlm_for_re",
            "pretrained": pretrained_backbone,  # backbone pretrained
        }
    }
    model = KieNet(model_config)
    return model


@register_model
def layoutxlm_ser(pretrained: bool = True, **kwargs) -> KieNet:
    pretrained_backbone = not pretrained
    model_config = {
        "type": "kie",
        "backbone": {
            "name": "layoutxlm_for_ser",
            "pretrained": pretrained_backbone,  # backbone pretrained
        },
    }
    model = KieNet(model_config)
    return model
