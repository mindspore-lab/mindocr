from ._registry import register_model
from .backbones.mindcv_models.utils import load_pretrained
from .base_model import BaseModel

__all__ = ["SVTR", "svtr_tiny", "svtr_tiny_ch"]


def _cfg(url="", input_size=(3, 32, 100), **kwargs):
    return {"url": url, "input_size": input_size, **kwargs}


default_cfgs = {
    "svtr_tiny": _cfg(
        url="https://download.mindspore.cn/toolkits/mindocr/svtr/svtr_tiny-8542b3bb.ckpt",
        input_size=(3, 64, 256),
    ),
    "svtr_tiny_ch": _cfg(
        url="https://download.mindspore.cn/toolkits/mindocr/svtr/svtr_tiny_ch-2ee6ade4.ckpt",
        input_size=(3, 32, 320),
    )
}


class SVTR(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def svtr_tiny(pretrained=False, **kwargs):
    model_config = {
        "transform": {
            "name": "STN_ON",
            "in_channels": 3,
            "tps_inputsize": [32, 64],
            "tps_outputsize": [32, 100],
            "num_control_points": 20,
            "tps_margins": [0.05, 0.05],
            "stn_activation": "none",
        },
        "backbone": {
            "name": "SVTRNet",
            "pretrained": False,
            "img_size": [32, 100],
            "out_channels": 192,
            "patch_merging": "Conv",
            "embed_dim": [64, 128, 256],
            "depth": [3, 6, 3],
            "num_heads": [2, 4, 8],
            "mixer": [
                "Local",
                "Local",
                "Local",
                "Local",
                "Local",
                "Local",
                "Global",
                "Global",
                "Global",
                "Global",
                "Global",
                "Global",
            ],
            "local_mixer": [[7, 11], [7, 11], [7, 11]],
            "last_stage": True,
            "prenorm": False,
        },
        "neck": {"name": "Img2Seq"},
        "head": {
            "name": "CTCHead",
            "out_channels": 37,
        },
    }

    model = SVTR(model_config)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs["svtr_tiny"]
        load_pretrained(model, default_cfg)

    return model


@register_model
def svtr_tiny_ch(pretrained=False, **kwargs):
    model_config = {
        "backbone": {
            "name": "SVTRNet",
            "pretrained": False,
            "img_size": [32, 320],
            "out_channels": 96,
            "patch_merging": "Conv",
            "embed_dim": [64, 128, 256],
            "depth": [3, 6, 3],
            "num_heads": [2, 4, 8],
            "mixer": [
                "Local",
                "Local",
                "Local",
                "Local",
                "Local",
                "Local",
                "Global",
                "Global",
                "Global",
                "Global",
                "Global",
                "Global",
            ],
            "local_mixer": [[7, 11], [7, 11], [7, 11]],
            "last_stage": True,
            "prenorm": False,
        },
        "neck": {"name": "Img2Seq"},
        "head": {
            "name": "CTCHead",
            "out_channels": 6624,
        },
    }

    model = SVTR(model_config)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs["svtr_tiny_ch"]
        load_pretrained(model, default_cfg)

    return model
