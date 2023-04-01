"""
MindSpore implementation of Xception.
Refer to Xception: Deep Learning with Depthwise Separable Convolutions.
"""

import mindspore.common.initializer as init
from mindspore import Tensor, nn, ops

from .layers import GlobalAvgPooling
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "Xception",
    "xception",
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
    "xception": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/xception/xception-2c1e711df.ckpt"),
}


class SeparableConv2d(nn.Cell):
    """SeparableCon2d module of Xception"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, group=in_channels, pad_mode="pad",
                               padding=padding)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, pad_mode="valid")

    def construct(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Cell):
    """Basic module of Xception"""

    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        reps: int,
        strides: int = 1,
        start_with_relu: bool = True,
        grow_first: bool = True,
    ):
        super().__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, pad_mode="valid", has_bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU()
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(nn.ReLU())
            rep.append(SeparableConv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for _ in range(reps - 1):
            rep.append(nn.ReLU())
            rep.append(SeparableConv2d(filters, filters, kernel_size=3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU())
            rep.append(SeparableConv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU()

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, pad_mode="same"))
        self.rep = nn.SequentialCell(*rep)

    def construct(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x = ops.add(x, skip)
        return x


class Xception(nn.Cell):
    r"""Xception model architecture from
    `"Deep Learning with Depthwise Separable Convolutions" <https://arxiv.org/abs/1610.02357>`_.

    Args:
        num_classes (int) : number of classification classes. Default: 1000.
        in_channels (int): number the channels of the input. Default: 3.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
        blocks = []
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, pad_mode="valid")
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, pad_mode="valid")
        self.bn2 = nn.BatchNorm2d(64)

        # Entry flow
        blocks.append(Block(64, 128, 2, 2, start_with_relu=False, grow_first=True))
        blocks.append(Block(128, 256, 2, 2, start_with_relu=True, grow_first=True))
        blocks.append(Block(256, 728, 2, 2, start_with_relu=True, grow_first=True))

        # Middle flow
        for _ in range(8):
            blocks.append(Block(728, 728, 3, 1, start_with_relu=True, grow_first=True))

        # Exit flow
        blocks.append(Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False))

        self.blocks = nn.SequentialCell(blocks)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.pool = GlobalAvgPooling()
        self.dropout = nn.Dropout()
        self.classifier = nn.Dense(2048, num_classes)

        self._initialize_weights()

    def forward_features(self, x: Tensor) -> Tensor:
        """forward the backbone of Xception"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.blocks(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
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

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(init.initializer(init.XavierUniform(), cell.weight.shape, cell.weight.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(init.Normal(0.01, 0), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(init.Constant(0), cell.bias.shape, cell.weight.dtype))


@register_model
def xception(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> Xception:
    """Get Xception model.
    Refer to the base class `models.Xception` for more details."""
    default_cfg = default_cfgs["xception"]
    model = Xception(num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
