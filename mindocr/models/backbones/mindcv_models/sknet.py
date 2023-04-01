"""
MindSpore implementation of `SKNet`.
Refer to Selective Kernel Networks.
"""

from typing import Dict, List, Optional, Type, Union

from mindspore import Tensor, nn

from .layers.selective_kernel import SelectiveKernel
from .registry import register_model
from .resnet import ResNet
from .utils import load_pretrained

__all__ = [
    "SKNet",
    "skresnet18",
    "skresnet34",
    "skresnet50",
    "skresnext50_32x4d",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "conv1",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "skresnet18": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/sknet/sknet18-868228e5.ckpt"),
    "skresnet34": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/sknet/skresnet34-d668b629.ckpt"),
    "skresnet50": _cfg(url=""),
    "skresnext50_32x4d": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/sknet/skresnext50_32x4d-395413a2.ckpt"),
}


class SelectiveKernelBasic(nn.Cell):
    """build basic block of sknet"""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        down_sample: Optional[nn.Cell] = None,
        base_width: int = 64,
        norm: Optional[nn.Cell] = None,
        sk_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d

        if sk_kwargs is None:
            sk_kwargs = {}

        assert groups == 1, "BasicBlock only supports cardinality of 1"
        assert base_width == 64, "BasicBlock doest not support changing base width"

        self.conv1 = SelectiveKernel(
            in_channels, out_channels, stride=stride, **sk_kwargs)
        self.conv2 = nn.SequentialCell([
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1, pad_mode="pad"),
            norm(out_channels * self.expansion)
        ])

        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)
        out += identity
        out = self.relu(out)
        return out


class SelectiveKernelBottleneck(nn.Cell):
    """build the bottleneck of the sknet"""

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        down_sample: Optional[nn.Cell] = None,
        groups: int = 1,
        base_width: int = 64,
        norm: Optional[nn.Cell] = None,
        sk_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d

        if sk_kwargs is None:
            sk_kwargs = {}

        width = int(out_channels * (base_width / 64.0)) * groups
        self.conv1 = nn.SequentialCell([
            nn.Conv2d(in_channels, width, kernel_size=1),
            norm(width)
        ])
        self.conv2 = SelectiveKernel(
            width, width, stride=stride, groups=groups, **sk_kwargs)
        self.conv3 = nn.SequentialCell([
            nn.Conv2d(width, out_channels * self.expansion, kernel_size=1),
            norm(out_channels * self.expansion)
        ])

        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.down_sample:
            identity = self.down_sample(x)
        out += identity
        out = self.relu(out)
        return out


class SKNet(ResNet):
    r"""SKNet model class, based on
    `"Selective Kernel Networks" <https://arxiv.org/abs/1903.06586>`_

    Args:
        block: block of sknet.
        layers: number of layers of each stage.
        num_classes: number of classification classes. Default: 1000.
        in_channels: number the channels of the input. Default: 3.
        groups: number of groups for group conv in blocks. Default: 1.
        base_width: base width of pre group hidden channel in blocks. Default: 64.
        norm: normalization layer in blocks. Default: None.
        sk_kwargs: kwargs of selective kernel. Default: None.
    """

    def __init__(
        self,
        block: Type[nn.Cell],
        layers: List[int],
        num_classes: int = 1000,
        in_channels: int = 3,
        groups: int = 1,
        base_width: int = 64,
        norm: Optional[nn.Cell] = None,
        sk_kwargs: Optional[Dict] = None,
    ) -> None:
        self.sk_kwargs: Optional[Dict] = sk_kwargs  # make pylint happy
        super().__init__(block, layers, num_classes, in_channels, groups, base_width, norm)

    def _make_layer(
        self,
        block: Type[Union[SelectiveKernelBasic, SelectiveKernelBottleneck]],
        channels: int,
        block_nums: int,
        stride: int = 1,
    ) -> nn.SequentialCell:
        down_sample = None

        if stride != 1 or self.input_channels != channels * block.expansion:
            down_sample = nn.SequentialCell([
                nn.Conv2d(self.input_channels, channels * block.expansion, kernel_size=1, stride=stride),
                self.norm(channels * block.expansion)
            ])

        layers = []
        layers.append(
            block(
                self.input_channels,
                channels,
                stride=stride,
                down_sample=down_sample,
                groups=self.groups,
                base_width=self.base_with,
                norm=self.norm,
                sk_kwargs=self.sk_kwargs,
            )
        )
        self.input_channels = channels * block.expansion

        for _ in range(1, block_nums):
            layers.append(
                block(
                    self.input_channels,
                    channels,
                    groups=self.groups,
                    base_width=self.base_with,
                    norm=self.norm,
                    sk_kwargs=self.sk_kwargs,
                )
            )

        return nn.SequentialCell(layers)


@register_model
def skresnet18(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ResNet:
    """Get 18 layers SKNet model.
    Refer to the base class `models.SKNet` for more details.
    """
    default_cfg = default_cfgs["skresnet18"]
    sk_kwargs = dict(rd_ratio=1 / 8, rd_divisor=16, split_input=True)
    model = SKNet(SelectiveKernelBasic, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels,
                  sk_kwargs=sk_kwargs, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def skresnet34(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ResNet:
    """Get 34 layers SKNet model.
    Refer to the base class `models.SKNet` for more details.
    """
    default_cfg = default_cfgs["skresnet34"]
    sk_kwargs = dict(rd_ratio=1 / 8, rd_divisor=16, split_input=True)
    model = SKNet(SelectiveKernelBasic, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels,
                  sk_kwargs=sk_kwargs, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def skresnet50(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ResNet:
    """Get 50 layers SKNet model.
    Refer to the base class `models.SKNet` for more details.
    """
    default_cfg = default_cfgs["skresnet50"]
    sk_kwargs = dict(split_input=True)
    model = SKNet(SelectiveKernelBottleneck, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels,
                  sk_kwargs=sk_kwargs, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def skresnext50_32x4d(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ResNet:
    """Get 50 layers SKNeXt model with 32 groups of GPConv.
    Refer to the base class `models.SKNet` for more details.
    """
    default_cfg = default_cfgs["skresnext50_32x4d"]
    sk_kwargs = dict(rd_ratio=1 / 16, rd_divisor=32, split_input=False)
    model = SKNet(SelectiveKernelBottleneck, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels,
                  sk_kwargs=sk_kwargs, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
