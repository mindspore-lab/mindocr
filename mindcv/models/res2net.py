"""
MindSpore implementation of `Res2Net`.
Refer to Res2Net: A New Multi-scale Backbone Architecture.
"""

import math
from typing import List, Optional, Type

import mindspore.common.initializer as init
from mindspore import Tensor, nn, ops

from .layers.pooling import GlobalAvgPooling
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "Res2Net",
    "res2net50",
    "res2net101",
    "res2net152",
    "res2net50_v1b",
    "res2net101_v1b",
    "res2net152_v1b",
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
    "res2net50": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/res2net/res2net50-f42cf71b.ckpt"),
    "res2net101": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/res2net/res2net101-8ae60132.ckpt"),
    "res2net152": _cfg(url=""),
    "res2net50_v1b": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/res2net/res2net50_v1b-99304e92.ckpt"),
    "res2net101_v1b": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/res2net/res2net101_v1b-7e6db001.ckpt"),
    "res2net152_v1b": _cfg(url=""),
}


class Bottle2neck(nn.Cell):
    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 26,
        scale: int = 4,
        stype: str = "normal",
        norm: Optional[nn.Cell] = None,
        down_sample: Optional[nn.Cell] = None,
    ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d

        width = int(math.floor(out_channels * (base_width / 64.0))) * groups

        self.conv1 = nn.Conv2d(in_channels, width * scale, kernel_size=1)
        self.bn1 = norm(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == "stage":
            self.pool = nn.SequentialCell([
                nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT"),
                nn.AvgPool2d(kernel_size=3, stride=stride),
            ])

        self.convs = nn.CellList()
        self.bns = nn.CellList()
        for _ in range(self.nums):
            self.convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, pad_mode="pad"))
            self.bns.append(norm(width))

        self.conv3 = nn.Conv2d(width * scale, out_channels * self.expansion,
                               kernel_size=1, stride=1)
        self.bn3 = norm(out_channels * self.expansion)

        self.relu = nn.ReLU()
        self.down_sample = down_sample
        self.stype = stype
        self.scale = scale
        self.width = width

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = ops.split(out, axis=1, output_num=self.scale)

        sp = self.convs[0](spx[0])
        sp = self.bns[0](sp)
        sp = self.relu(sp)
        out = sp

        for i in range(1, self.nums):
            if self.stype == "stage":
                sp = spx[i]
            else:
                sp = sp[:, :, :, :]
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.bns[i](sp)
            sp = self.relu(sp)

            out = ops.concat((out, sp), axis=1)

        if self.scale != 1 and self.stype == "normal":
            out = ops.concat((out, spx[self.nums]), axis=1)
        elif self.scale != 1 and self.stype == "stage":
            out = ops.concat((out, self.pool(spx[self.nums])), axis=1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class Res2Net(nn.Cell):
    r"""Res2Net model class, based on
    `"Res2Net: A New Multi-scale Backbone Architecture" <https://arxiv.org/abs/1904.01169>`_

    Args:
        block: block of resnet.
        layer_nums: number of layers of each stage.
        version: variety of Res2Net, 'res2net' or 'res2net_v1b'. Default: 'res2net'.
        num_classes: number of classification classes. Default: 1000.
        in_channels: number the channels of the input. Default: 3.
        groups: number of groups for group conv in blocks. Default: 1.
        base_width: base width of pre group hidden channel in blocks. Default: 26.
        scale: scale factor of Bottle2neck. Default: 4.
        norm: normalization layer in blocks. Default: None.
    """

    def __init__(
        self,
        block: Type[nn.Cell],
        layer_nums: List[int],
        version: str = "res2net",
        num_classes: int = 1000,
        in_channels: int = 3,
        groups: int = 1,
        base_width: int = 26,
        scale=4,
        norm: Optional[nn.Cell] = None,
    ) -> None:
        super().__init__()
        assert version in ["res2net", "res2net_v1b"]
        self.version = version

        if norm is None:
            norm = nn.BatchNorm2d
        self.norm = norm

        self.num_classes = num_classes
        self.input_channels = 64
        self.groups = groups
        self.base_width = base_width
        self.scale = scale
        if self.version == "res2net":
            self.conv1 = nn.Conv2d(in_channels, self.input_channels, kernel_size=7,
                                   stride=2, padding=3, pad_mode="pad")
        elif self.version == "res2net_v1b":
            self.conv1 = nn.SequentialCell([
                nn.Conv2d(in_channels, self.input_channels // 2, kernel_size=3,
                          stride=2, padding=1, pad_mode="pad"),
                norm(self.input_channels // 2),
                nn.ReLU(),
                nn.Conv2d(self.input_channels // 2, self.input_channels // 2, kernel_size=3,
                          stride=1, padding=1, pad_mode="pad"),
                norm(self.input_channels // 2),
                nn.ReLU(),
                nn.Conv2d(self.input_channels // 2, self.input_channels, kernel_size=3,
                          stride=1, padding=1, pad_mode="pad"),
            ])

        self.bn1 = norm(self.input_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.SequentialCell([
            nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT"),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ])
        self.layer1 = self._make_layer(block, 64, layer_nums[0])
        self.layer2 = self._make_layer(block, 128, layer_nums[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layer_nums[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layer_nums[3], stride=2)

        self.pool = GlobalAvgPooling()
        self.num_features = 512 * block.expansion
        self.classifier = nn.Dense(self.num_features, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    init.initializer(init.HeNormal(math.sqrt(5), mode="fan_out", nonlinearity="relu"),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer(init.HeUniform(math.sqrt(5), mode="fan_in", nonlinearity="leaky_relu"),
                                         cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(init.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(init.initializer("zeros", cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(init.HeUniform(math.sqrt(5), mode="fan_in", nonlinearity="leaky_relu"),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def _make_layer(
        self,
        block: Type[nn.Cell],
        channels: int,
        block_nums: int,
        stride: int = 1,
    ) -> nn.SequentialCell:
        down_sample = None

        if stride != 1 or self.input_channels != channels * block.expansion:
            if stride == 1 or self.version == "res2net":
                down_sample = nn.SequentialCell([
                    nn.Conv2d(self.input_channels, channels * block.expansion, kernel_size=1, stride=stride),
                    self.norm(channels * block.expansion)
                ])
            else:
                down_sample = nn.SequentialCell([
                    nn.AvgPool2d(kernel_size=stride, stride=stride, pad_mode="same"),
                    nn.Conv2d(self.input_channels, channels * block.expansion, kernel_size=1, stride=1),
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
                base_width=self.base_width,
                scale=self.scale,
                stype="stage",
                norm=self.norm,
            )
        )
        self.input_channels = channels * block.expansion

        for _ in range(1, block_nums):
            layers.append(
                block(
                    self.input_channels,
                    channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    scale=self.scale,
                    norm=self.norm,
                )
            )

        return nn.SequentialCell(layers)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def res2net50(pretrained: bool = False, num_classes: int = 1001, in_channels=3, **kwargs):
    """Get 50 layers Res2Net model.
    Refer to the base class `models.Res2Net` for more details.
    """
    default_cfg = default_cfgs["res2net50"]
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def res2net101(pretrained: bool = False, num_classes: int = 1001, in_channels=3, **kwargs):
    """Get 101 layers Res2Net model.
    Refer to the base class `models.Res2Net` for more details.
    """
    default_cfg = default_cfgs["res2net101"]
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def res2net152(pretrained: bool = False, num_classes: int = 1001, in_channels=3, **kwargs):
    """Get 152 layers Res2Net model.
    Refer to the base class `models.Res2Net` for more details.
    """
    default_cfg = default_cfgs["res2net152"]
    model = Res2Net(Bottle2neck, [3, 8, 36, 3], num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def res2net50_v1b(pretrained: bool = False, num_classes: int = 1001, in_channels=3, **kwargs):
    default_cfg = default_cfgs["res2net50_v1b"]
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], version="res2net_v1b", num_classes=num_classes,
                    in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def res2net101_v1b(pretrained: bool = False, num_classes: int = 1001, in_channels=3, **kwargs):
    default_cfg = default_cfgs["res2net101_v1b"]
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], version="res2net_v1b", num_classes=num_classes,
                    in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def res2net152_v1b(pretrained: bool = False, num_classes: int = 1001, in_channels=3, **kwargs):
    default_cfg = default_cfgs["res2net152_v1b"]
    model = Res2Net(Bottle2neck, [3, 8, 36, 3], version="res2net_v1b", num_classes=num_classes,
                    in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
