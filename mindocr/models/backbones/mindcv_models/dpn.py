"""
MindSpore implementation of `DPN`.
Refer to: Dual Path Networks
"""

import math
from collections import OrderedDict
from typing import Tuple

import mindspore.common.initializer as init
from mindspore import Tensor, nn, ops

from .layers.pooling import GlobalAvgPooling
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "DPN",
    "dpn92",
    "dpn98",
    "dpn131",
    "dpn107",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "features.conv1.conv",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "dpn92": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/dpn/dpn92-e3e0fca.ckpt"),
    "dpn98": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/dpn/dpn98-119a8207.ckpt"),
    "dpn107": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/dpn/dpn107-7d7df07b.ckpt"),
    "dpn131": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/dpn/dpn131-47f084b3.ckpt"),
}


class BottleBlock(nn.Cell):
    """A block for the Dual Path Architecture"""

    def __init__(
        self,
        in_channel: int,
        num_1x1_a: int,
        num_3x3_b: int,
        num_1x1_c: int,
        inc: int,
        g: int,
        key_stride: int,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channel, eps=1e-3, momentum=0.9)
        self.conv1 = nn.Conv2d(in_channel, num_1x1_a, 1, stride=1)
        self.bn2 = nn.BatchNorm2d(num_1x1_a, eps=1e-3, momentum=0.9)
        self.conv2 = nn.Conv2d(num_1x1_a, num_3x3_b, 3, key_stride, pad_mode="pad", padding=1, group=g)
        self.bn3 = nn.BatchNorm2d(num_3x3_b, eps=1e-3, momentum=0.9)
        self.conv3_r = nn.Conv2d(num_3x3_b, num_1x1_c, 1, stride=1)
        self.conv3_d = nn.Conv2d(num_3x3_b, inc, 1, stride=1)

        self.relu = nn.ReLU()

    def construct(self, x: Tensor):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        return (self.conv3_r(x), self.conv3_d(x))


class DualPathBlock(nn.Cell):
    """A block for Dual Path Networks to combine proj, residual and densely network"""

    def __init__(
        self,
        in_channel: int,
        num_1x1_a: int,
        num_3x3_b: int,
        num_1x1_c: int,
        inc: int,
        g: int,
        _type: str = "normal",
        cat_input: bool = True,
    ):
        super().__init__()
        self.num_1x1_c = num_1x1_c

        if _type == "proj":
            key_stride = 1
            self.has_proj = True
        if _type == "down":
            key_stride = 2
            self.has_proj = True
        if _type == "normal":
            key_stride = 1
            self.has_proj = False

        self.cat_input = cat_input

        if self.has_proj:
            self.c1x1_w_bn = nn.BatchNorm2d(in_channel, eps=1e-3, momentum=0.9)
            self.c1x1_w_relu = nn.ReLU()
            self.c1x1_w_r = nn.Conv2d(in_channel, num_1x1_c, kernel_size=1, stride=key_stride,
                                      pad_mode="pad", padding=0)
            self.c1x1_w_d = nn.Conv2d(in_channel, 2 * inc, kernel_size=1, stride=key_stride,
                                      pad_mode="pad", padding=0)

        self.layers = BottleBlock(in_channel, num_1x1_a, num_3x3_b, num_1x1_c, inc, g, key_stride)

    def construct(self, x: Tensor):
        if self.cat_input:
            data_in = ops.concat(x, axis=1)
        else:
            data_in = x

        if self.has_proj:
            data_o = self.c1x1_w_bn(data_in)
            data_o = self.c1x1_w_relu(data_o)
            data_o1 = self.c1x1_w_r(data_o)
            data_o2 = self.c1x1_w_d(data_o)
        else:
            data_o1 = x[0]
            data_o2 = x[1]

        out = self.layers(data_in)
        summ = ops.add(data_o1, out[0])
        dense = ops.concat((data_o2, out[1]), axis=1)
        return (summ, dense)


class DPN(nn.Cell):
    r"""DPN model class, based on
    `"Dual Path Networks" <https://arxiv.org/pdf/1707.01629.pdf>`_

    Args:
        num_init_channel: int type, the output channel of first blocks. Default: 64.
        k_r: int type, the first channel of each stage. Default: 96.
        g: int type,number of group in the conv2d. Default: 32.
        k_sec Tuple[int]: multiplicative factor for number of bottleneck layers. Default: 4.
        inc_sec Tuple[int]: the first output channel in each stage. Default: (16, 32, 24, 128).
        in_channels: int type, number of input channels. Default: 3.
        num_classes: int type, number of classification classes. Default: 1000.
    """

    def __init__(
        self,
        num_init_channel: int = 64,
        k_r: int = 96,
        g: int = 32,
        k_sec: Tuple[int, int, int, int] = (3, 4, 20, 3),
        inc_sec: Tuple[int, int, int, int] = (16, 32, 24, 128),
        in_channels: int = 3,
        num_classes: int = 1000,
    ):
        super().__init__()
        blocks = OrderedDict()

        # conv1
        blocks["conv1"] = nn.SequentialCell(OrderedDict([
            ("conv", nn.Conv2d(in_channels, num_init_channel, kernel_size=7, stride=2, pad_mode="pad", padding=3)),
            ("norm", nn.BatchNorm2d(num_init_channel, eps=1e-3, momentum=0.9)),
            ("relu", nn.ReLU()),
            ("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")),
        ]))

        # conv2
        bw = 256
        inc = inc_sec[0]
        r = int((k_r * bw) / 256)
        blocks["conv2_1"] = DualPathBlock(num_init_channel, r, r, bw, inc, g, "proj", False)
        in_channel = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks[f"conv2_{i}"] = DualPathBlock(in_channel, r, r, bw, inc, g, "normal")
            in_channel += inc

        # conv3
        bw = 512
        inc = inc_sec[1]
        r = int((k_r * bw) / 256)
        blocks["conv3_1"] = DualPathBlock(in_channel, r, r, bw, inc, g, "down")
        in_channel = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks[f"conv3_{i}"] = DualPathBlock(in_channel, r, r, bw, inc, g, "normal")
            in_channel += inc

        # conv4
        bw = 1024
        inc = inc_sec[2]
        r = int((k_r * bw) / 256)
        blocks["conv4_1"] = DualPathBlock(in_channel, r, r, bw, inc, g, "down")
        in_channel = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks[f"conv4_{i}"] = DualPathBlock(in_channel, r, r, bw, inc, g, "normal")
            in_channel += inc

        # conv5
        bw = 2048
        inc = inc_sec[3]
        r = int((k_r * bw) / 256)
        blocks["conv5_1"] = DualPathBlock(in_channel, r, r, bw, inc, g, "down")
        in_channel = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks[f"conv5_{i}"] = DualPathBlock(in_channel, r, r, bw, inc, g, "normal")
            in_channel += inc

        self.features = nn.SequentialCell(blocks)
        self.conv5_x = nn.SequentialCell(OrderedDict([
            ("norm", nn.BatchNorm2d(in_channel, eps=1e-3, momentum=0.9)),
            ("relu", nn.ReLU()),
        ]))
        self.avgpool = GlobalAvgPooling()
        self.classifier = nn.Dense(in_channel, num_classes)
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

    def forward_feature(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = ops.concat(x, axis=1)
        x = self.conv5_x(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_feature(x)
        x = self.forward_head(x)
        return x


@register_model
def dpn92(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> DPN:
    """Get 92 layers DPN model.
     Refer to the base class `models.DPN` for more details."""
    default_cfg = default_cfgs["dpn92"]
    model = DPN(num_init_channel=64, k_r=96, g=32, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
                num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def dpn98(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> DPN:
    """Get 98 layers DPN model.
     Refer to the base class `models.DPN` for more details."""
    default_cfg = default_cfgs["dpn98"]
    model = DPN(num_init_channel=96, k_r=160, g=40, k_sec=(3, 6, 20, 3), inc_sec=(16, 32, 32, 128),
                num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def dpn131(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> DPN:
    """Get 131 layers DPN model.
     Refer to the base class `models.DPN` for more details."""
    default_cfg = default_cfgs["dpn131"]
    model = DPN(num_init_channel=128, k_r=160, g=40, k_sec=(4, 8, 28, 3), inc_sec=(16, 32, 32, 128),
                num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def dpn107(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> DPN:
    """Get 107 layers DPN model.
     Refer to the base class `models.DPN` for more details."""
    default_cfg = default_cfgs["dpn107"]
    model = DPN(num_init_channel=128, k_r=200, g=50, k_sec=(4, 8, 20, 3), inc_sec=(20, 64, 64, 128),
                num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
