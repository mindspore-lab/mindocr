from ._registry import register_model
from .backbones.mindcv_models.utils import load_pretrained
from .base_model import BaseModel

__all__ = ["Master", "master_resnet31"]


def _cfg(url="", input_size=(3, 32, 100), **kwargs):
    return {"url": url, "input_size": input_size, **kwargs}


default_cfgs = {
    "master_resnet31": _cfg(
        url="https://download.mindspore.cn/toolkits/mindocr/master/master_resnet31-e7bfbc97.ckpt",
        input_size=(3, 48, 160),
    ),
}


class Master(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def master_resnet31(pretrained=False, **kwargs):
    model_config = {
        "backbone": {
            "name": "rec_resnet_master_resnet31",
            "pretrained": False
        },
        "neck": {
            "name": "MasterEncoder",
            "with_encoder": "False"
        },
        "head": {
            "name": "MasterDecoder",
            "out_channels": 94,
            "batch_max_length": 30
        }
    }

    model = Master(model_config)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs["master_resnet31"]
        load_pretrained(model, default_cfg)

    return model
