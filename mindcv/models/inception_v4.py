"""
MindSpore implementation of `InceptionV4`.
Refer to Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning.
"""

from typing import Tuple, Union

import mindspore.common.initializer as init
from mindspore import Tensor, nn, ops

from .layers.pooling import GlobalAvgPooling
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "InceptionV4",
    "inception_v4",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "features.0.conv2d_1a_3x3.conv",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "inception_v4": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/inception_v4/inception_v4-db9c45b3.ckpt")
}


class BasicConv2d(nn.Cell):
    """A block for conv bn and relu"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple] = 1,
        stride: int = 1,
        padding: int = 0,
        pad_mode: str = "same",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=padding, pad_mode=pad_mode)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9997)
        self.relu = nn.ReLU()

    def construct(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Stem(nn.Cell):
    """Inception V4 model blocks."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv2d_1a_3x3 = BasicConv2d(in_channels, 32, kernel_size=3, stride=2, pad_mode="valid")
        self.conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, stride=1, pad_mode="valid")
        self.conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, stride=1, pad_mode="pad", padding=1)

        self.mixed_3a_branch_0 = nn.MaxPool2d(3, stride=2)
        self.mixed_3a_branch_1 = BasicConv2d(64, 96, kernel_size=3, stride=2, pad_mode="valid")

        self.mixed_4a_branch_0 = nn.SequentialCell([
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, pad_mode="valid")
        ])

        self.mixed_4a_branch_1 = nn.SequentialCell([
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 64, kernel_size=(1, 7), stride=1),
            BasicConv2d(64, 64, kernel_size=(7, 1), stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, pad_mode="valid")
        ])

        self.mixed_5a_branch_0 = BasicConv2d(192, 192, kernel_size=3, stride=2, pad_mode="valid")
        self.mixed_5a_branch_1 = nn.MaxPool2d(3, stride=2)

    def construct(self, x: Tensor) -> Tensor:
        x = self.conv2d_1a_3x3(x)  # 149 x 149 x 32
        x = self.conv2d_2a_3x3(x)  # 147 x 147 x 32
        x = self.conv2d_2b_3x3(x)  # 147 x 147 x 64

        x0 = self.mixed_3a_branch_0(x)
        x1 = self.mixed_3a_branch_1(x)
        x = ops.concat((x0, x1), axis=1)  # 73 x 73 x 160

        x0 = self.mixed_4a_branch_0(x)
        x1 = self.mixed_4a_branch_1(x)
        x = ops.concat((x0, x1), axis=1)  # 71 x 71 x 192

        x0 = self.mixed_5a_branch_0(x)
        x1 = self.mixed_5a_branch_1(x)
        x = ops.concat((x0, x1), axis=1)  # 35 x 35 x 384
        return x


class InceptionA(nn.Cell):
    """Inception V4 model basic architecture"""

    def __init__(self) -> None:
        super().__init__()
        self.branch_0 = BasicConv2d(384, 96, kernel_size=1, stride=1)
        self.branch_1 = nn.SequentialCell([
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, pad_mode="pad", padding=1)
        ])
        self.branch_2 = nn.SequentialCell([
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, pad_mode="pad", padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, pad_mode="pad", padding=1)
        ])
        self.branch_3 = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, stride=1, pad_mode="same"),
            BasicConv2d(384, 96, kernel_size=1, stride=1)
        ])

    def construct(self, x: Tensor) -> Tensor:
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x4 = ops.concat((x0, x1, x2, x3), axis=1)
        return x4


class InceptionB(nn.Cell):
    """Inception V4 model basic architecture"""

    def __init__(self) -> None:
        super().__init__()
        self.branch_0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)
        self.branch_1 = nn.SequentialCell([
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1, 7), stride=1),
            BasicConv2d(224, 256, kernel_size=(7, 1), stride=1),
        ])
        self.branch_2 = nn.SequentialCell([
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(7, 1), stride=1),
            BasicConv2d(192, 224, kernel_size=(1, 7), stride=1),
            BasicConv2d(224, 224, kernel_size=(7, 1), stride=1),
            BasicConv2d(224, 256, kernel_size=(1, 7), stride=1)
        ])
        self.branch_3 = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, stride=1, pad_mode="same"),
            BasicConv2d(1024, 128, kernel_size=1, stride=1)
        ])

    def construct(self, x: Tensor) -> Tensor:
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x4 = ops.concat((x0, x1, x2, x3), axis=1)
        return x4


class ReductionA(nn.Cell):
    """Inception V4 model Residual Connections"""

    def __init__(self) -> None:
        super().__init__()
        self.branch_0 = BasicConv2d(384, 384, kernel_size=3, stride=2, pad_mode="valid")
        self.branch_1 = nn.SequentialCell([
            BasicConv2d(384, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=3, stride=1, pad_mode="pad", padding=1),
            BasicConv2d(224, 256, kernel_size=3, stride=2, pad_mode="valid"),
        ])
        self.branch_2 = nn.MaxPool2d(3, stride=2)

    def construct(self, x: Tensor) -> Tensor:
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = ops.concat((x0, x1, x2), axis=1)
        return x3


class ReductionB(nn.Cell):
    """Inception V4 model Residual Connections"""

    def __init__(self) -> None:
        super().__init__()
        self.branch_0 = nn.SequentialCell([
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2, pad_mode="valid"),
        ])
        self.branch_1 = nn.SequentialCell([
            BasicConv2d(1024, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=(1, 7), stride=1),
            BasicConv2d(256, 320, kernel_size=(7, 1), stride=1),
            BasicConv2d(320, 320, kernel_size=3, stride=2, pad_mode="valid")
        ])
        self.branch_2 = nn.MaxPool2d(3, stride=2)

    def construct(self, x: Tensor) -> Tensor:
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = ops.concat((x0, x1, x2), axis=1)
        return x3  # 8 x 8 x 1536


class InceptionC(nn.Cell):
    """Inception V4 model basic architecture"""

    def __init__(self) -> None:
        super().__init__()
        self.branch_0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)

        self.branch_1 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch_1_1 = BasicConv2d(384, 256, kernel_size=(1, 3), stride=1)
        self.branch_1_2 = BasicConv2d(384, 256, kernel_size=(3, 1), stride=1)

        self.branch_2 = nn.SequentialCell([
            BasicConv2d(1536, 384, kernel_size=1, stride=1),
            BasicConv2d(384, 448, kernel_size=(3, 1), stride=1),
            BasicConv2d(448, 512, kernel_size=(1, 3), stride=1),
        ])
        self.branch_2_1 = BasicConv2d(512, 256, kernel_size=(1, 3), stride=1)
        self.branch_2_2 = BasicConv2d(512, 256, kernel_size=(3, 1), stride=1)

        self.branch_3 = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, stride=1, pad_mode="same"),
            BasicConv2d(1536, 256, kernel_size=1, stride=1)
        ])

    def construct(self, x: Tensor) -> Tensor:
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x1_1 = self.branch_1_1(x1)
        x1_2 = self.branch_1_2(x1)
        x1 = ops.concat((x1_1, x1_2), axis=1)
        x2 = self.branch_2(x)
        x2_1 = self.branch_2_1(x2)
        x2_2 = self.branch_2_2(x2)
        x2 = ops.concat((x2_1, x2_2), axis=1)
        x3 = self.branch_3(x)
        return ops.concat((x0, x1, x2, x3), axis=1)


class InceptionV4(nn.Cell):
    r"""Inception v4 model architecture from
    `"Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning" <https://arxiv.org/abs/1602.07261>`_.  # noqa: E501

    Args:
        num_classes: number of classification classes. Default: 1000.
        in_channels: number the channels of the input. Default: 3.
        drop_rate: dropout rate of the layer before main classifier. Default: 0.2.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
        drop_rate: float = 0.2,
    ) -> None:
        super().__init__()
        blocks = [Stem(in_channels)]
        for _ in range(4):
            blocks.append(InceptionA())
        blocks.append(ReductionA())
        for _ in range(7):
            blocks.append(InceptionB())
        blocks.append(ReductionB())
        for _ in range(3):
            blocks.append(InceptionC())
        self.features = nn.SequentialCell(blocks)

        self.pool = GlobalAvgPooling()
        self.dropout = nn.Dropout(1 - drop_rate)
        self.num_features = 1536
        self.classifier = nn.Dense(self.num_features, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    init.initializer(init.XavierUniform(), cell.weight.shape, cell.weight.dtype))

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.features(x)
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
def inception_v4(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> InceptionV4:
    """Get InceptionV4 model.
    Refer to the base class `models.InceptionV4` for more details."""
    default_cfg = default_cfgs["inception_v4"]
    model = InceptionV4(num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
