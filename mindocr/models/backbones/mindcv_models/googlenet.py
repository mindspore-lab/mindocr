"""
MindSpore implementation of `GoogLeNet`.
Refer to Going deeper with convolutions.
"""

import math
from typing import Tuple, Union

import mindspore.common.initializer as init
from mindspore import Tensor, nn, ops

from .layers.pooling import GlobalAvgPooling
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "GoogLeNet",
    "googlenet",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "conv1.conv",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "googlenet": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/googlenet/googlenet-5552fcd3.ckpt"),
}


class BasicConv2d(nn.Cell):
    """A block for combine conv and relu"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        pad_mode: str = "same",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=padding, pad_mode=pad_mode)
        self.relu = nn.ReLU()

    def construct(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.relu(x)
        return x


class Inception(nn.Cell):
    """Inception module of GoogLeNet."""

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
    ) -> None:
        super().__init__()
        self.b1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.b2 = nn.SequentialCell([
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3),
        ])
        self.b3 = nn.SequentialCell([
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5),
        ])
        self.b4 = nn.SequentialCell([
            nn.MaxPool2d(kernel_size=3, stride=1, pad_mode="same"),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        ])

    def construct(self, x: Tensor) -> Tensor:
        branch1 = self.b1(x)
        branch2 = self.b2(x)
        branch3 = self.b3(x)
        branch4 = self.b4(x)
        return ops.concat((branch1, branch2, branch3, branch4), axis=1)


class InceptionAux(nn.Cell):
    """Inception module for the aux classifier head"""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        drop_rate: float = 0.7,
    ) -> None:
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Dense(2048, 1024)
        self.fc2 = nn.Dense(1024, num_classes)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(1 - drop_rate)

    def construct(self, x: Tensor) -> Tensor:
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Cell):
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <https://arxiv.org/abs/1409.4842>`_.

    Args:
        num_classes: number of classification classes. Default: 1000.
        aux_logits: use auxiliary classifier or not. Default: False.
        in_channels: number the channels of the input. Default: 3.
        drop_rate: dropout rate of the layer before main classifier. Default: 0.2.
        drop_rate_aux: dropout rate of the layer before auxiliary classifier. Default: 0.7.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = False,
        in_channels: int = 3,
        drop_rate: float = 0.2,
        drop_rate_aux: float = 0.7,
    ) -> None:
        super().__init__()
        self.aux_logits = aux_logits
        self.conv1 = BasicConv2d(in_channels, 64, kernel_size=7, stride=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes, drop_rate=drop_rate_aux)
            self.aux2 = InceptionAux(528, num_classes, drop_rate=drop_rate_aux)

        self.pool = GlobalAvgPooling()
        self.dropout = nn.Dropout(keep_prob=1 - drop_rate)
        self.classifier = nn.Dense(1024, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(init.initializer(init.HeNormal(0, mode='fan_in', nonlinearity='leaky_relu'),
                                                      cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(init.Constant(0), cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d) or isinstance(cell, nn.BatchNorm1d):
                cell.gamma.set_data(init.initializer(init.Constant(1), cell.gamma.shape, cell.gamma.dtype))
                if cell.beta is not None:
                    cell.beta.set_data(init.initializer(init.Constant(0), cell.beta.shape, cell.gamma.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(init.HeUniform(math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu'),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(init.Constant(0), cell.bias.shape, cell.weight.dtype))

    def construct(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)
        else:
            aux1 = None

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)
        else:
            aux2 = None

        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.pool(x)
        x = self.dropout(x)
        x = self.classifier(x)

        if self.aux_logits and self.training:
            return x, aux2, aux1
        return x


@register_model
def googlenet(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> GoogLeNet:
    """Get GoogLeNet model.
    Refer to the base class `models.GoogLeNet` for more details."""
    default_cfg = default_cfgs["googlenet"]
    model = GoogLeNet(num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
