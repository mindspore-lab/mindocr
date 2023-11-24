from ._registry import register_model
from .base_model import BaseModel


class KieNet(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)
        self.use_visual_backbone = config["backbone"]["use_visual_backbone"]

    def construct(self, x):
        if self.use_visual_backbone is True:
            image = x[4]
        else:
            image = None

        x = self.backbone(
            input_ids=x[0],
            bbox=x[1],
            attention_mask=x[2],
            token_type_ids=x[3],
            image=image,
            position_ids=None,
            head_mask=None,
            labels=None,
        )
        x = self.head(x)

        return x


@register_model
def layoutxlm_ser(pretrained: bool = True, use_visual_backbone: bool = True, use_float16: bool = False, **kwargs):
    model_config = {
        "type": "kie",
        "backbone": {
            "name": "layoutxlm",
            "pretrained": pretrained,  # backbone pretrained
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

    return model


@register_model
def vi_layoutxlm_ser(pretrained: bool = True, use_visual_backbone: bool = False, use_float16: bool = False, **kwargs):
    model_config = {
        "type": "kie",
        "backbone": {
            "name": "layoutxlm",
            "pretrained": pretrained,  # backbone pretrained
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

    return model
