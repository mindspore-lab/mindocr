from ._registry import register_model
from .backbones.mindcv_models.utils import load_pretrained
from .base_model import BaseModel

__all__ = ["RobustScanner", "robustscanner_resnet31"]


def _cfg(url='', input_size=(3, 48, 160), **kwargs):
    return {
        'url': url,
        'input_size': input_size,
        **kwargs
    }


default_cfgs = {
    "robustscanner_resnet31": _cfg(
        url="https://download.mindspore.cn/toolkits/mindocr/robustscanner/robustscanner_resnet31-f27eab37.ckpt",
    ),
}


class RobustScanner(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def robustscanner_resnet31(pretrained=False, **kwargs):
    model_config = {
        "backbone": {
            'name': 'rec_resnet31',
            'pretrained': False
        },
        "head": {
            "name": 'RobustScannerHead',
            "out_channels": 93,
            "enc_outchannles": 128,
            "hybrid_dec_rnn_layers": 2,
            "hybrid_dec_dropout": 0.,
            "position_dec_rnn_layers": 2,
            "start_idx": 91,
            "mask": True,
            "padding_idx": 92,
            "encode_value": False,
            "max_text_len": 40
        }
    }
    model = RobustScanner(model_config)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['robustscanner_resnet31']
        load_pretrained(model, default_cfg)

    return model
