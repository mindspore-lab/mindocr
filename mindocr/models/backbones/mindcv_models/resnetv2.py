"""
MindSpore implementation of `ResNetV2`.
Refer to Identity Mappings in Deep Residual Networks.
"""

from typing import Optional

from mindspore import Tensor, nn

from .registry import register_model
from .resnet import ResNet
from .utils import load_pretrained

__all__ = [
    "resnetv2_50",
    "resnetv2_101",
]


def _cfg(url='', **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "conv1",
        "classifier": "classifier",
        **kwargs
    }


default_cfgs = {
    "resnetv2_50": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnetv2/resnetv2_50-3c2f143b.ckpt"),
    "resnetv2_101": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnetv2/resnetv2_101-5d4c49a1.ckpt"),
}


class PreActBottleneck(nn.Cell):
    expansion: int = 4

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int = 1,
                 groups: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None
                 ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d

        width = int(channels * (base_width / 64.0)) * groups

        self.bn1 = norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1)

        self.bn2 = norm(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, pad_mode='pad', group=groups)

        self.bn3 = norm(width)
        self.conv3 = nn.Conv2d(width, channels * self.expansion,
                               kernel_size=1, stride=1)

        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.relu(out)

        residual = out

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.down_sample is not None:
            identity = self.down_sample(residual)

        out += identity

        return out


@register_model
def resnetv2_50(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 50 layers ResNetV2 model.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs['resnetv2_50']
    model = ResNet(PreActBottleneck, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def resnetv2_101(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 101 layers ResNetV2 model.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["resnetv2_101"]
    model = ResNet(PreActBottleneck, [3, 4, 23, 3], num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
