from ._registry import register_model
from .backbones.mindcv_models.utils import load_pretrained
from .base_model import BaseModel


def _cfg(url="", **kwargs):
    return {"url": url, **kwargs}


default_cfgs = {
    "yolov8": _cfg(
        url="https://download.mindspore.cn/toolkits/mindocr/yolov8/yolov8n-4b9e8004.ckpt"
    ),
}


class LayoutNet(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def yolov8(pretrained=False, pretrained_backbone=True, **kwargs):
    model_config = {
        "backbone": {
            "name": "yolov8_backbone",
            "depth_multiple": 0.33,
            "width_multiple": 0.25,
            "max_channels": 1024,
            "nc": 5,
            "stride": [8, 16, 32, 64],
            "reg_max": 16,
            "sync_bn": False,
            "out_channels": [64, 128, 192, 256],
        },
        "neck": {
            "name": "YOLOv8Neck",
            "index": [20, 23, 26, 29],
        },
        "head": {
            "name": "YOLOv8Head",
            "nc": 5,
            "reg_max": 16,
            "stride": [8, 16, 32, 64],
            "sync_bn": False,
        },
    }
    model = LayoutNet(model_config)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['yolov8']
        load_pretrained(model, default_cfg)

    return model
