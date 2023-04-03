"""MindSpore implementation of `GhostNet`."""

import math

import numpy as np

from mindspore import Tensor, nn, ops

from .layers.pooling import GlobalAvgPooling
from .layers.squeeze_excite import SqueezeExcite
from .registry import register_model
from .utils import load_pretrained, make_divisible

__all__ = [
    "GhostNet",
    "ghostnet_1x",
    "ghostnet_nose_1x",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "conv_stem",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "ghostnet_1x": _cfg(url=""),
    "ghostnet_nose_1x": _cfg(url=""),
}


class GhostGate(nn.Cell):
    """Implementation for (relu6 + 3) / 6"""

    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU6()

    def construct(self, x):
        return self.relu6(x + 3.0) * 0.16666667


class ConvBnAct(nn.Cell):
    """A block for conv bn and relu"""

    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride,
                              padding=kernel_size // 2, pad_mode="pad", has_bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Cell):
    def __init__(
        self,
        inp: int,
        oup: int,
        kernel_size: int = 1,
        ratio: int = 2,
        dw_size: int = 3,
        stride: int = 1,
        relu: bool = True,
    ) -> None:
        super().__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.SequentialCell(
            nn.Conv2d(inp, init_channels, kernel_size, stride=stride,
                      padding=kernel_size // 2, pad_mode="pad", has_bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU() if relu else nn.SequentialCell(),
        )

        self.cheap_operation = nn.SequentialCell(
            nn.Conv2d(init_channels, new_channels, dw_size, stride=1,
                      padding=dw_size // 2, pad_mode="pad", group=init_channels, has_bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU() if relu else nn.SequentialCell(),
        )

    def construct(self, x: Tensor) -> Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = ops.concat((x1, x2), axis=1)

        return out[:, : self.oup, :, :]


class GhostBottleneck(nn.Cell):
    def __init__(
        self,
        in_chs: int,
        mid_chs: int,
        out_chs: int,
        dw_kernel_size: int = 3,
        stride: int = 1,
        se_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2, pad_mode="pad",
                                     group=mid_chs, has_bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, rd_ratio=1.0 / 4, rd_divisor=4, act_layer=nn.ReLU, gate_layer=GhostGate)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.SequentialCell()
        else:
            self.shortcut = nn.SequentialCell(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          pad_mode="pad", padding=(dw_kernel_size - 1) // 2, group=in_chs, has_bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, has_bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def construct(self, x: Tensor) -> Tensor:
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class GhostNet(nn.Cell):
    r"""GhostNet model class, based on
    `"GhostNet: More Features from Cheap Operations " <https://arxiv.org/abs/1911.11907>`_

    Args:
        cfgs: the config of the GhostNet.
        num_classes: number of classification classes. Default: 1000.
        in_channels: number of input channels. Default: 3.
        width: base width of hidden channel in blocks. Default: 1.0
        droupout: the probability of the features before classification. Default: 0.2
    """

    def __init__(
        self,
        cfgs,
        num_classes: int = 1000,
        in_channels: int = 3,
        width: float = 1.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout_rate = dropout

        # building first layer
        output_channel = make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(in_channels, output_channel, kernel_size=3,
                                   padding=1, stride=2, has_bias=False, pad_mode="pad")
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU()
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        exp_size = 128
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = make_divisible(c * width, 4)
                hidden_channel = make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s, se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.SequentialCell([*layers]))

        output_channel = make_divisible(exp_size * width, 4)
        stages.append(nn.SequentialCell([ConvBnAct(input_channel, output_channel, 1)]))
        input_channel = output_channel

        self.blocks = nn.SequentialCell([*stages])

        # building last several layers
        output_channel = 1280
        self.global_pool = GlobalAvgPooling(keep_dims=True)
        self.conv_head = nn.Conv2d(input_channel, output_channel, kernel_size=1,
                                   padding=0, stride=1, has_bias=True, pad_mode="pad")
        self.act2 = nn.ReLU()
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Dense(output_channel, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.set_data(Tensor(np.random.normal(0, np.sqrt(2.0 / n), m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(Tensor(np.zeros(m.bias.data.shape, dtype="float32")))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
                m.beta.set_data(Tensor(np.zeros(m.beta.data.shape, dtype="float32")))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(Tensor(np.random.normal(0, 0.01, m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(Tensor(np.zeros(m.bias.data.shape, dtype="float32")))

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.conv_head(x)
        x = self.act2(x)
        x = ops.flatten(x)
        if self.dropout_rate > 0.0:
            x = self.dropout(x)
        x = self.classifier(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


model_cfgs = {
    "1x": {
        "cfg": [
            # k, t, c, SE, s
            # stage1
            [[3, 16, 16, 0, 1]],
            # stage2
            [[3, 48, 24, 0, 2]],
            [[3, 72, 24, 0, 1]],
            # stage3
            [[5, 72, 40, 0.25, 2]],
            [[5, 120, 40, 0.25, 1]],
            # stage4
            [[3, 240, 80, 0, 2]],
            [[3, 200, 80, 0, 1],
             [3, 184, 80, 0, 1],
             [3, 184, 80, 0, 1],
             [3, 480, 112, 0.25, 1],
             [3, 672, 112, 0.25, 1]],
            # stage5
            [[5, 672, 160, 0.25, 2]],
            [[5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1],
             [5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1]]
        ],
        "cls_ch_squeeze": 960,
        "cls_ch_expand": 1280,
    },
    "nose_1x": {
        "cfg": [
            # k, exp, c,  se,     nl,  s,
            # stage1
            [[3, 16, 16, 0, 1]],
            # stage2
            [[3, 48, 24, 0, 2]],
            [[3, 72, 24, 0, 1]],
            # stage3
            [[5, 72, 40, 0, 2]],
            [[5, 120, 40, 0, 1]],
            # stage4
            [[3, 240, 80, 0, 2]],
            [[3, 200, 80, 0, 1],
             [3, 184, 80, 0, 1],
             [3, 184, 80, 0, 1],
             [3, 480, 112, 0, 1],
             [3, 672, 112, 0, 1]],
            # stage5
            [[5, 672, 160, 0, 2]],
            [[5, 960, 160, 0, 1],
             [5, 960, 160, 0, 1],
             [5, 960, 160, 0, 1],
             [5, 960, 160, 0, 1]]
        ],
        "cls_ch_squeeze": 960,
        "cls_ch_expand": 1280,
    },
}


@register_model
def ghostnet_1x(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> GhostNet:
    """Get GhostNet model.
    Refer to the base class 'models.GhostNet' for more details.
    """
    model_args = model_cfgs["1x"]["cfg"]
    model = GhostNet(cfgs=model_args, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, model_args, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def ghostnet_nose_1x(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> GhostNet:
    """Get GhostNet model without SEModule.
    Refer to the base class 'models.GhostNet' for more details.
    """
    model_args = model_cfgs["nose_1x"]["cfg"]
    model = GhostNet(cfgs=model_args, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, model_args, num_classes=num_classes, in_channels=in_channels)

    return model
