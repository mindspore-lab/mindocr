from addict import Dict
import mindspore.nn as nn
from mindspore.ops import operations as ops
from .backbones import build_backbone
from .necks import build_neck
from .heads import build_head
from ._registry import register_model

__all__ = ['CRNN']


class CRNN(nn.Cell):
    def __init__(self, config: dict):
        """
        Args:
            config (dict): model config 
        """
        super(CRNN, self).__init__()

        config = Dict(config)
        backbone_name = config.backbone.pop('name')
        self.backbone = build_backbone(backbone_name, **config.backbone)

        neck_name = config.neck.pop('name')
        self.neck = build_neck(
            neck_name, in_channels=self.backbone.out_channels, **config.neck)

        head_name = config.head.pop('name')
        self.head = build_head(
            head_name, in_channels=self.neck.out_channels, **config.head)

        self.model_name = f'{backbone_name}_{neck_name}_{head_name}'

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def construct(self, x):
        x = self.backbone(x)
        bs, fs, _, _ = x.shape
        x = self.reshape(x, (bs, fs, -1))
        x = self.transpose(x, (0, 2, 1))
        x = self.neck(x)
        x = self.head(x)
        return x


@register_model
def crnn_r34(pretrained=False, **kwargs):
    model_config = {
        "backbone": {
            'name': 'rec_resnet34',
            'pretrained': False
        },
        "neck": {
            "name": 'BiLSTM',
            "out_channels": 512,
            "hidden_size": 256,
            "batch_size": 8,
            "num_layers": 2,
            "dropout": 0.0,
            "bidirectional": True,
        },
        "head": {
            "name": 'CTCHead',
            "out_channels": 37,
        }
    }
    model = CRNN(model_config)

    # load pretrained weights
    if pretrained:
        raise NotImplementedError()

    return model
