"""
MindSpore implementation of `SqueezeNet`.
Refer to SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size.
"""

import mindspore.common.initializer as init
from mindspore import Tensor, nn, ops

from .layers.pooling import GlobalAvgPooling
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "SqueezeNet",
    "squeezenet1_0",
    "squeezenet1_1",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "features.0",
        "classifier": "classifier.1",
        **kwargs,
    }


default_cfgs = {
    "squeezenet1_0": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/squeezenet/squeezenet1_0-e2d78c4a.ckpt"),
    "squeezenet1_1": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/squeezenet/squeezenet1_1-da256d3a.ckpt"),
}


class Fire(nn.Cell):
    """define the basic block of squeezenet"""

    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
        expand1x1_channels: int,
        expand3x3_channels: int,
    ) -> None:
        super().__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, has_bias=True)
        self.squeeze_activation = nn.ReLU()
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1, has_bias=True)
        self.expand1x1_activation = nn.ReLU()
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, pad_mode="same", has_bias=True)
        self.expand3x3_activation = nn.ReLU()

    def construct(self, x: Tensor) -> Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return ops.concat((self.expand1x1_activation(self.expand1x1(x)),
                           self.expand3x3_activation(self.expand3x3(x))), axis=1)


class SqueezeNet(nn.Cell):
    r"""SqueezeNet model class, based on
    `"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size" <https://arxiv.org/abs/1602.07360>`_  # noqa: E501

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 227 x 227, so ensure your images are sized accordingly.

    Args:
        version: version of the architecture, '1_0' or '1_1'. Default: '1_0'.
        num_classes: number of classification classes. Default: 1000.
        drop_rate: dropout rate of the classifier. Default: 0.5.
        in_channels: number the channels of the input. Default: 3.
    """

    def __init__(
        self,
        version: str = "1_0",
        num_classes: int = 1000,
        drop_rate: float = 0.5,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if version == "1_0":
            self.features = nn.SequentialCell([
                nn.Conv2d(in_channels, 96, kernel_size=7, stride=2, pad_mode="valid", has_bias=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2),
                Fire(512, 64, 256, 256),
            ])
        elif version == "1_1":
            self.features = nn.SequentialCell([
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, pad_mode="pad", has_bias=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            ])
        else:
            raise ValueError(f"Unsupported SqueezeNet version {version}: 1_0 or 1_1 expected")

        self.final_conv = nn.Conv2d(512, num_classes, kernel_size=1, has_bias=True)
        self.classifier = nn.SequentialCell([
            nn.Dropout(keep_prob=1 - drop_rate),
            self.final_conv,
            nn.ReLU(),
            GlobalAvgPooling()
        ])
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                if cell is self.final_conv:
                    cell.weight.set_data(init.initializer(init.Normal(), cell.weight.shape, cell.weight.dtype))
                else:
                    cell.weight.set_data(init.initializer(init.HeUniform(), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.classifier(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def squeezenet1_0(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> SqueezeNet:
    """Get SqueezeNet model of version 1.0.
    Refer to the base class `models.SqueezeNet` for more details.
    """
    default_cfg = default_cfgs["squeezenet1_0"]
    model = SqueezeNet(version="1_0", num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def squeezenet1_1(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> SqueezeNet:
    """Get SqueezeNet model of version 1.1.
    Refer to the base class `models.SqueezeNet` for more details.
    """
    default_cfg = default_cfgs["squeezenet1_1"]
    model = SqueezeNet(version="1_1", num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
