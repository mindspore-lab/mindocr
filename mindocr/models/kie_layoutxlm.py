from ._registry import register_model
from .backbones.mindcv_models.utils import load_pretrained
from .base_model import BaseModel


def _cfg(url="", **kwargs):
    return {"url": url, **kwargs}


default_cfgs = {
    "layoutxlm": _cfg(
        url="https://download.mindspore.cn/toolkits/mindocr/layoutxlm/ser_layoutxlm_base-a4ea148e.ckpt"
    ),
    "vi_layoutxlm": _cfg(
        url="https://download.mindspore.cn/toolkits/mindocr/vi-layoutxlm/ser_vi_layoutxlm-f3c83585.ckpt"
    ),
}


class KieNet(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)
        self.use_visual_backbone = config["backbone"]["use_visual_backbone"]

    def construct(self, x):
        if self.use_visual_backbone is True:
            image = x[4]
        else:
            image = None

        out = self.backbone(
            input_ids=x[0],
            bbox=x[1],
            attention_mask=x[2],
            token_type_ids=x[3],
            image=image,
            position_ids=None,
            head_mask=None,
        )
        out = self.head(out, input_id=x[0])

        return out


@register_model
def layoutxlm_ser(
    pretrained: bool = True,
    pretrained_backbone=False,
    use_visual_backbone: bool = True,
    use_float16: bool = False,
    **kwargs
):
    model_config = {
        "type": "kie",
        "backbone": {
            "name": "layoutxlm",
            "pretrained": pretrained_backbone,  # backbone pretrained
            "use_visual_backbone": use_visual_backbone,
            "use_float16": use_float16,
        },
        "head": {
            "name": "TokenClassificationHead",
            "num_classes": 7,
            "use_visual_backbone": use_visual_backbone,
            "use_float16": use_float16,
            "dropout_prod": None,
        },
    }
    model = KieNet(model_config)
    if pretrained:
        default_cfg = default_cfgs["layoutxlm"]
        load_pretrained(model, default_cfg)

    return model


@register_model
def vi_layoutxlm_ser(
    pretrained: bool = True,
    pretrained_backbone: bool = False,
    use_visual_backbone: bool = False,
    use_float16: bool = False,
    **kwargs
):
    model_config = {
        "type": "kie",
        "backbone": {
            "name": "layoutxlm",
            "pretrained": pretrained_backbone,  # backbone pretrained
            "use_visual_backbone": use_visual_backbone,
            "use_float16": use_float16,
        },
        "head": {
            "name": "TokenClassificationHead",
            "num_classes": 7,
            "use_visual_backbone": use_visual_backbone,
            "use_float16": use_float16,
            "dropout_prod": None,
        },
    }
    model = KieNet(model_config)
    if pretrained:
        default_cfg = default_cfgs["vi_layoutxlm"]
        load_pretrained(model, default_cfg)

    return model
