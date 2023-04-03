"""
MindSpore implementation of `MixNet`.
Refer to MixConv: Mixed Depthwise Convolutional Kernels
"""

import math
from typing import Optional

import mindspore.common.initializer as init
from mindspore import Tensor, nn, ops

from .layers.pooling import GlobalAvgPooling
from .layers.squeeze_excite import SqueezeExcite
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "MixNet",
    "mixnet_s",
    "mixnet_m",
    "mixnet_l",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "stem_conv.0", "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "mixnet_s": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/mixnet/mixnet_s-2a5ef3a3.ckpt"),
    "mixnet_m": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/mixnet/mixnet_m-74cc4cb1.ckpt"),
    "mixnet_l": _cfg(url=""),
}


def _roundchannels(filters: float, divisor: int = 8, min_depth: Optional[int] = None) -> int:
    if min_depth is None:
        min_depth = divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return new_filters


def _splitchannels(channels: int, num_groups: int) -> list:
    split_channels = [channels // num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels


class Swish(nn.Cell):
    def __init__(self) -> None:
        super(Swish, self).__init__()
        self.sigmoid = ops.Sigmoid()

    def construct(self, x: Tensor) -> Tensor:
        return x * self.sigmoid(x)


class GroupedConv2d(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: list,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super(GroupedConv2d, self).__init__()
        self.num_groups = len(kernel_size)
        if self.num_groups == 1:
            self.grouped_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size[0],
                stride=stride,
                pad_mode="pad",
                padding=padding,
                has_bias=False
            )
        else:
            self.split_in_channels = _splitchannels(in_channels, self.num_groups)
            self.split_out_channels = _splitchannels(out_channels, self.num_groups)

            self.grouped_conv = nn.CellList()
            for i in range(self.num_groups):
                self.grouped_conv.append(nn.Conv2d(
                    self.split_in_channels[i],
                    self.split_out_channels[i],
                    kernel_size[i],
                    stride=stride,
                    pad_mode="pad",
                    padding=padding,
                    has_bias=False
                ))

    def construct(self, x: Tensor) -> Tensor:
        if self.num_groups == 1:
            return self.grouped_conv(x)

        output = []
        start, end = 0, 0
        for i in range(self.num_groups):
            start, end = end, end + self.split_in_channels[i]
            x_split = x[:, start:end]

            conv = self.grouped_conv[i]
            output.append(conv(x_split))

        return ops.concat(output, axis=1)


class MDConv(nn.Cell):
    """Mixed Depth-wise Convolution"""

    def __init__(self, channels: int, kernel_size: list, stride: int) -> None:
        super(MDConv, self).__init__()
        self.num_groups = len(kernel_size)

        if self.num_groups == 1:
            self.mixed_depthwise_conv = nn.Conv2d(
                channels,
                channels,
                kernel_size[0],
                stride=stride,
                pad_mode="pad",
                padding=kernel_size[0] // 2,
                group=channels,
                has_bias=False
            )
        else:
            self.split_channels = _splitchannels(channels, self.num_groups)

            self.mixed_depthwise_conv = nn.CellList()
            for i in range(self.num_groups):
                self.mixed_depthwise_conv.append(nn.Conv2d(
                    self.split_channels[i],
                    self.split_channels[i],
                    kernel_size[i],
                    stride=stride,
                    pad_mode="pad",
                    padding=kernel_size[i] // 2,
                    group=self.split_channels[i],
                    has_bias=False
                ))

    def construct(self, x: Tensor) -> Tensor:
        if self.num_groups == 1:
            return self.mixed_depthwise_conv(x)

        output = []
        start, end = 0, 0
        for i in range(self.num_groups):
            start, end = end, end + self.split_channels[i]
            x_split = x[:, start:end]

            conv = self.mixed_depthwise_conv[i]
            output.append(conv(x_split))

        return ops.concat(output, axis=1)


class MixNetBlock(nn.Cell):
    """Basic Block of MixNet"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: list = [3],
        expand_ksize: list = [1],
        project_ksize: list = [1],
        stride: int = 1,
        expand_ratio: int = 1,
        activation: str = "ReLU",
        se_ratio: float = 0.0,
    ) -> None:
        super(MixNetBlock, self).__init__()
        assert activation in ["ReLU", "Swish"]
        self.activation = Swish if activation == "Swish" else nn.ReLU

        expand_channels = in_channels * expand_ratio
        self.residual_connection = (stride == 1 and in_channels == out_channels)

        conv = []
        if expand_ratio != 1:
            # expand
            conv.extend([
                GroupedConv2d(in_channels, expand_channels, expand_ksize),
                nn.BatchNorm2d(expand_channels),
                self.activation()
            ])

        # depthwise
        conv.extend([
            MDConv(expand_channels, kernel_size, stride),
            nn.BatchNorm2d(expand_channels),
            self.activation()
        ])

        if se_ratio > 0:
            squeeze_channels = int(in_channels * se_ratio)
            squeeze_excite = SqueezeExcite(expand_channels, rd_channels=squeeze_channels)
            conv.append(squeeze_excite)

        # projection phase
        conv.extend([
            GroupedConv2d(expand_channels, out_channels, project_ksize),
            nn.BatchNorm2d(out_channels)
        ])

        self.convs = nn.SequentialCell(conv)

    def construct(self, x: Tensor) -> Tensor:
        if self.residual_connection:
            return x + self.convs(x)
        else:
            return self.convs(x)


class MixNet(nn.Cell):
    r"""MixNet model class, based on
    `"MixConv: Mixed Depthwise Convolutional Kernels" <https://arxiv.org/abs/1907.09595>`_

    Args:
        arch: size of the architecture. "small", "medium" or "large". Default: "small".
        num_classes: number of classification classes. Default: 1000.
        in_channels: number of the channels of the input. Default: 3.
        feature_size: numbet of the channels of the output features. Default: 1536.
        drop_rate: rate of dropout for classifier. Default: 0.2.
        depth_multiplier: expansion coefficient of channels. Default: 1.0.
    """

    def __init__(
        self,
        arch: str = "small",
        num_classes: int = 1000,
        in_channels: int = 3,
        feature_size: int = 1536,
        drop_rate: float = 0.2,
        depth_multiplier: float = 1.0
    ) -> None:
        super(MixNet, self).__init__()
        if arch == "small":
            block_configs = [
                [16, 16, [3], [1], [1], 1, 1, "ReLU", 0.0],
                [16, 24, [3], [1, 1], [1, 1], 2, 6, "ReLU", 0.0],
                [24, 24, [3], [1, 1], [1, 1], 1, 3, "ReLU", 0.0],
                [24, 40, [3, 5, 7], [1], [1], 2, 6, "Swish", 0.5],
                [40, 40, [3, 5], [1, 1], [1, 1], 1, 6, "Swish", 0.5],
                [40, 40, [3, 5], [1, 1], [1, 1], 1, 6, "Swish", 0.5],
                [40, 40, [3, 5], [1, 1], [1, 1], 1, 6, "Swish", 0.5],
                [40, 80, [3, 5, 7], [1], [1, 1], 2, 6, "Swish", 0.25],
                [80, 80, [3, 5], [1], [1, 1], 1, 6, "Swish", 0.25],
                [80, 80, [3, 5], [1], [1, 1], 1, 6, "Swish", 0.25],
                [80, 120, [3, 5, 7], [1, 1], [1, 1], 1, 6, "Swish", 0.5],
                [120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, "Swish", 0.5],
                [120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, "Swish", 0.5],
                [120, 200, [3, 5, 7, 9, 11], [1], [1], 2, 6, "Swish", 0.5],
                [200, 200, [3, 5, 7, 9], [1], [1, 1], 1, 6, "Swish", 0.5],
                [200, 200, [3, 5, 7, 9], [1], [1, 1], 1, 6, "Swish", 0.5]
            ]
            stem_channels = 16
            drop_rate = drop_rate
        else:
            block_configs = [
                [24, 24, [3], [1], [1], 1, 1, "ReLU", 0.0],
                [24, 32, [3, 5, 7], [1, 1], [1, 1], 2, 6, "ReLU", 0.0],
                [32, 32, [3], [1, 1], [1, 1], 1, 3, "ReLU", 0.0],
                [32, 40, [3, 5, 7, 9], [1], [1], 2, 6, "Swish", 0.5],
                [40, 40, [3, 5], [1, 1], [1, 1], 1, 6, "Swish", 0.5],
                [40, 40, [3, 5], [1, 1], [1, 1], 1, 6, "Swish", 0.5],
                [40, 40, [3, 5], [1, 1], [1, 1], 1, 6, "Swish", 0.5],
                [40, 80, [3, 5, 7], [1], [1], 2, 6, "Swish", 0.25],
                [80, 80, [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, "Swish", 0.25],
                [80, 80, [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, "Swish", 0.25],
                [80, 80, [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, "Swish", 0.25],
                [80, 120, [3], [1], [1], 1, 6, "Swish", 0.5],
                [120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, "Swish", 0.5],
                [120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, "Swish", 0.5],
                [120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, "Swish", 0.5],
                [120, 200, [3, 5, 7, 9], [1], [1], 2, 6, "Swish", 0.5],
                [200, 200, [3, 5, 7, 9], [1], [1, 1], 1, 6, "Swish", 0.5],
                [200, 200, [3, 5, 7, 9], [1], [1, 1], 1, 6, "Swish", 0.5],
                [200, 200, [3, 5, 7, 9], [1], [1, 1], 1, 6, "Swish", 0.5]
            ]
            if arch == "medium":
                stem_channels = 24
                drop_rate = drop_rate
            elif arch == "large":
                stem_channels = 24
                depth_multiplier *= 1.3
                drop_rate = drop_rate
            else:
                raise ValueError(f"Unsupported model type {arch}")

        if depth_multiplier != 1.0:
            stem_channels = _roundchannels(stem_channels * depth_multiplier)

            for i, conf in enumerate(block_configs):
                conf_ls = list(conf)
                conf_ls[0] = _roundchannels(conf_ls[0] * depth_multiplier)
                conf_ls[1] = _roundchannels(conf_ls[1] * depth_multiplier)
                block_configs[i] = tuple(conf_ls)

        # stem convolution
        self.stem_conv = nn.SequentialCell([
            nn.Conv2d(in_channels, stem_channels, 3, stride=2, pad_mode="pad", padding=1),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU()
        ])

        # building MixNet blocks
        layers = []
        for inc, outc, k, ek, pk, s, er, ac, se in block_configs:
            layers.append(MixNetBlock(
                inc,
                outc,
                kernel_size=k,
                expand_ksize=ek,
                project_ksize=pk,
                stride=s,
                expand_ratio=er,
                activation=ac,
                se_ratio=se
            ))
        self.layers = nn.SequentialCell(layers)

        # head
        self.head_conv = nn.SequentialCell([
            nn.Conv2d(block_configs[-1][1], feature_size, 1, pad_mode="pad", padding=0),
            nn.BatchNorm2d(feature_size),
            nn.ReLU()
        ])

        self.pool = GlobalAvgPooling()
        self.dropout = nn.Dropout(keep_prob=1 - drop_rate)
        self.classifier = nn.Dense(feature_size, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                fan_out = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                cell.weight.set_data(
                    init.initializer(init.Normal(math.sqrt(2.0 / fan_out)),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer("zeros", cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(init.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(init.initializer("zeros", cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(init.Uniform(1.0 / math.sqrt(cell.weight.shape[0])),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.head_conv(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def mixnet_s(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["mixnet_s"]
    model = MixNet(arch="small", in_channels=in_channels, num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mixnet_m(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["mixnet_m"]
    model = MixNet(arch="medium", in_channels=in_channels, num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mixnet_l(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["mixnet_l"]
    model = MixNet(arch="large", in_channels=in_channels, num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
