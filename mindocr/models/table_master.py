from ._registry import register_model
from .backbones.mindcv_models.utils import load_pretrained
from .base_model import BaseModel


def _cfg(url="", **kwargs):
    return {"url": url, **kwargs}


default_cfgs = {
    "table_master": _cfg(
        url="https://download-mindspore.osinfra.cn/toolkits/mindocr/tablemaster/table_master-78bf35bb.ckpt"
    ),
}


class TableMaster(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def table_master(
    pretrained: bool = True,
    **kwargs
):
    model_config = {
        "type": "table",
        "transform": None,
        "backbone": {
            "name": "table_resnet_extra",
            "gcb_config": {
                "ratio": 0.0625,
                "headers": 1,
                "att_scale": False,
                "fusion_type": "channel_add",
                "layers": [False, True, True, True],
            },
            "layers": [1, 2, 5, 3],
        },
        "head": {
            "name": "TableMasterHead",
            "out_channels": 43,
            "hidden_size": 512,
            "headers": 8,
            "dropout": 0.0,
            "d_ff": 2024,
            "max_text_length": 500,
            "loc_reg_num": 4
        },
    }
    model = TableMaster(model_config)
    if pretrained:
        default_cfg = default_cfgs["table_master"]
        load_pretrained(model, default_cfg)

    return model
