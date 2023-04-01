"""
MindSpore implementation of `ReXNet`.
Refer to ReXNet: Rethinking Channel Dimensions for Efficient Model Design.
"""
import math
from math import ceil
from typing import Any

import mindspore.common.initializer as init
import mindspore.nn as nn

from .layers import Conv2dNormActivation, DropPath, GlobalAvgPooling, SqueezeExcite
from .registry import register_model
from .utils import load_pretrained, make_divisible

__all__ = [
    "ReXNetV1",
    "rexnet_x09",
    "rexnet_x10",
    "rexnet_x13",
    "rexnet_x15",
    "rexnet_x20",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "first_conv": "",
        "classifier": "",
        **kwargs,
    }


default_cfgs = {
    "rexnet_x09": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/rexnet/rexnet_09-da498331.ckpt"),
    "rexnet_x10": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/rexnet/rexnet_10-c5fb2dc7.ckpt"),
    "rexnet_x13": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/rexnet/rexnet_13-a49c41e5.ckpt"),
    "rexnet_x15": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/rexnet/rexnet_15-37a931d3.ckpt"),
    "rexnet_x20": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/rexnet/rexnet_20-c5810914.ckpt"),
}


class LinearBottleneck(nn.Cell):
    """LinearBottleneck"""

    def __init__(
        self,
        in_channels,
        out_channels,
        exp_ratio,
        stride,
        use_se=True,
        se_ratio=1 / 12,
        ch_div=1,
        act_layer=nn.SiLU,
        dw_act_layer=nn.ReLU6,
        drop_path=None,
        **kwargs,
    ):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels <= out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        if exp_ratio != 1:
            dw_channels = in_channels * exp_ratio
            self.conv_exp = Conv2dNormActivation(in_channels, dw_channels, 1, activation=act_layer)
        else:
            dw_channels = in_channels
            self.conv_exp = None

        self.conv_dw = Conv2dNormActivation(dw_channels, dw_channels, 3, stride, padding=1,
                                            groups=dw_channels, activation=None)

        if use_se:
            self.se = SqueezeExcite(dw_channels,
                                    rd_channels=make_divisible(int(dw_channels * se_ratio), ch_div),
                                    norm=nn.BatchNorm2d)
        else:
            self.se = None
        self.act_dw = dw_act_layer()

        self.conv_pwl = Conv2dNormActivation(dw_channels, out_channels, 1, padding=0, activation=None)
        self.drop_path = drop_path

    def construct(self, x):
        shortcut = x
        if self.conv_exp is not None:
            x = self.conv_exp(x)
        x = self.conv_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.act_dw(x)
        x = self.conv_pwl(x)
        if self.use_shortcut:
            if self.drop_path is not None:
                x = self.drop_path(x)
            x[:, 0:self.in_channels] += shortcut
        return x


class ReXNetV1(nn.Cell):
    r"""ReXNet model class, based on
    `"Rethinking Channel Dimensions for Efficient Model Design" <https://arxiv.org/abs/2007.00992>`_

    Args:
        in_channels (int): number of the input channels. Default: 3.
        fi_channels (int): number of the final channels. Default: 180.
        initial_channels (int): initialize inplanes. Default: 16.
        width_mult (float): The ratio of the channel. Default: 1.0.
        depth_mult (float): The ratio of num_layers. Default: 1.0.
        num_classes (int) : number of classification classes. Default: 1000.
        use_se (bool): use SENet in LinearBottleneck. Default: True.
        se_ratio: (float): SENet reduction ratio. Default 1/12.
        drop_rate (float): dropout ratio. Default: 0.2.
        ch_div (int): divisible by ch_div. Default: 1.
        act_layer (nn.Cell): activation function in ConvNormAct. Default: nn.SiLU.
        dw_act_layer (nn.Cell): activation function after dw_conv. Default: nn.ReLU6.
        cls_useconv (bool): use conv in classification. Default: False.
    """

    def __init__(
        self,
        in_channels=3,
        fi_channels=180,
        initial_channels=16,
        width_mult=1.0,
        depth_mult=1.0,
        num_classes=1000,
        use_se=True,
        se_ratio=1 / 12,
        drop_rate=0.2,
        drop_path_rate=0.0,
        ch_div=1,
        act_layer=nn.SiLU,
        dw_act_layer=nn.ReLU6,
        cls_useconv=False,
    ):
        super(ReXNetV1, self).__init__()

        layers = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]
        use_ses = [False, False, True, True, True, True]

        layers = [ceil(element * depth_mult) for element in layers]
        strides = sum([[element] + [1] * (layers[idx] - 1)
                       for idx, element in enumerate(strides)], [])
        if use_se:
            use_ses = sum([[element] * layers[idx] for idx, element in enumerate(use_ses)], [])
        else:
            use_ses = [False] * sum(layers[:])
        exp_ratios = [1] * layers[0] + [6] * sum(layers[1:])

        self.depth = sum(layers[:]) * 3
        stem_channel = 32 / width_mult if width_mult < 1.0 else 32
        inplanes = initial_channels / width_mult if width_mult < 1.0 else initial_channels

        features = []
        in_channels_group = []
        out_channels_group = []

        for i in range(self.depth // 3):
            if i == 0:
                in_channels_group.append(int(round(stem_channel * width_mult)))
                out_channels_group.append(int(round(inplanes * width_mult)))
            else:
                in_channels_group.append(int(round(inplanes * width_mult)))
                inplanes += fi_channels / (self.depth // 3 * 1.0)
                out_channels_group.append(int(round(inplanes * width_mult)))

        stem_chs = make_divisible(round(stem_channel * width_mult), divisor=ch_div)
        self.stem = Conv2dNormActivation(in_channels, stem_chs, stride=2, padding=1, activation=act_layer)

        feat_chs = [stem_chs]
        feature_info = []
        curr_stride = 2
        features = []
        num_blocks = len(in_channels_group)
        for block_idx, (in_c, out_c, exp_ratio, stride, use_se) in enumerate(
            zip(in_channels_group, out_channels_group, exp_ratios, strides, use_ses)
        ):
            if stride > 1:
                fname = "stem" if block_idx == 0 else f"features.{block_idx - 1}"
                feature_info += [dict(num_chs=feat_chs[-1], reduction=curr_stride, module=fname)]
                curr_stride *= stride
            block_dpr = drop_path_rate * block_idx / (num_blocks - 1)  # stochastic depth linear decay rule
            drop_path = DropPath(block_dpr) if block_dpr > 0. else None
            features.append(LinearBottleneck(in_channels=in_c,
                                             out_channels=out_c,
                                             exp_ratio=exp_ratio,
                                             stride=stride,
                                             use_se=use_se,
                                             se_ratio=se_ratio,
                                             act_layer=act_layer,
                                             dw_act_layer=dw_act_layer,
                                             drop_path=drop_path))

        pen_channels = make_divisible(int(1280 * width_mult), divisor=ch_div)
        features.append(Conv2dNormActivation(out_channels_group[-1],
                                             pen_channels,
                                             kernel_size=1,
                                             activation=act_layer))

        features.append(GlobalAvgPooling(keep_dims=True))
        self.useconv = cls_useconv
        self.features = nn.SequentialCell(*features)
        if self.useconv:
            self.cls = nn.SequentialCell(
                nn.Dropout(1.0 - drop_rate),
                nn.Conv2d(pen_channels, num_classes, 1, has_bias=True))
        else:
            self.cls = nn.SequentialCell(
                nn.Dropout(1.0 - drop_rate),
                nn.Dense(pen_channels, num_classes))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, (nn.Conv2d, nn.Dense)):
                cell.weight.set_data(
                    init.initializer(init.HeUniform(math.sqrt(5), mode="fan_in", nonlinearity="relu"),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer(init.HeUniform(math.sqrt(5), mode="fan_in", nonlinearity="leaky_relu"),
                                         [1, cell.bias.shape[0]], cell.bias.dtype).reshape((-1)))

    def forward_features(self, x):
        x = self.stem(x)
        x = self.features(x)
        return x

    def forward_head(self, x):
        if not self.useconv:
            x = x.reshape((x.shape[0], -1))
            x = self.cls(x)
        else:
            x = self.cls(x).reshape((x.shape[0], -1))
        return x

    def construct(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _rexnet(
    arch: str,
    width_mult: float,
    in_channels: int,
    num_classes: int,
    pretrained: bool,
    **kwargs: Any,
) -> ReXNetV1:
    """ReXNet architecture."""
    default_cfg = default_cfgs[arch]
    model = ReXNetV1(width_mult=width_mult, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def rexnet_x09(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ReXNetV1:
    """Get ReXNet model with width multiplier of 0.9.
    Refer to the base class `models.ReXNetV1` for more details.
    """
    return _rexnet("rexnet_x09", 0.9, in_channels, num_classes, pretrained, **kwargs)


@register_model
def rexnet_x10(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ReXNetV1:
    """Get ReXNet model with width multiplier of 1.0.
    Refer to the base class `models.ReXNetV1` for more details.
    """
    return _rexnet("rexnet_x10", 1.0, in_channels, num_classes, pretrained, **kwargs)


@register_model
def rexnet_x13(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ReXNetV1:
    """Get ReXNet model with width multiplier of 1.3.
    Refer to the base class `models.ReXNetV1` for more details.
    """
    return _rexnet("rexnet_x13", 1.3, in_channels, num_classes, pretrained, **kwargs)


@register_model
def rexnet_x15(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ReXNetV1:
    """Get ReXNet model with width multiplier of 1.5.
    Refer to the base class `models.ReXNetV1` for more details.
    """
    return _rexnet("rexnet_x15", 1.5, in_channels, num_classes, pretrained, **kwargs)


@register_model
def rexnet_x20(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ReXNetV1:
    """Get ReXNet model with width multiplier of 2.0.
    Refer to the base class `models.ReXNetV1` for more details.
    """
    return _rexnet("rexnet_x20", 2.0, in_channels, num_classes, pretrained, **kwargs)
