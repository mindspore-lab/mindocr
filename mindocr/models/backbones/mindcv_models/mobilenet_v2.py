"""
MindSpore implementation of `MobileNetV2`.
Refer to MobileNetV2: Inverted Residuals and Linear Bottlenecks.
"""

import math

import mindspore.common.initializer as init
from mindspore import Tensor, nn

from .layers.pooling import GlobalAvgPooling
from .registry import register_model
from .utils import load_pretrained, make_divisible

__all__ = [
    "MobileNetV2",
    "mobilenet_v2_140_224",
    "mobilenet_v2_130_224",
    "mobilenet_v2_100_224",
    "mobilenet_v2_100_192",
    "mobilenet_v2_100_160",
    "mobilenet_v2_100_128",
    "mobilenet_v2_100_96",
    "mobilenet_v2_075_224",
    "mobilenet_v2_075_192",
    "mobilenet_v2_075_160",
    "mobilenet_v2_075_128",
    "mobilenet_v2_075_96",
    "mobilenet_v2_050_224",
    "mobilenet_v2_050_192",
    "mobilenet_v2_050_160",
    "mobilenet_v2_050_128",
    "mobilenet_v2_050_96",
    "mobilenet_v2_035_224",
    "mobilenet_v2_035_192",
    "mobilenet_v2_035_160",
    "mobilenet_v2_035_128",
    "mobilenet_v2_035_96",
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
    "mobilenet_v2_1.4_224": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/mobilenet/mobilenetv2/mobilenet_v2_140-015cfb04.ckpt"
    ),
    "mobilenet_v2_1.3_224": _cfg(
        url=""
    ),
    "mobilenet_v2_1.0_224": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/mobilenet/mobilenetv2/mobilenet_v2_100-52122156.ckpt"
    ),
    "mobilenet_v2_1.0_192": _cfg(
        url=""
    ),
    "mobilenet_v2_1.0_160": _cfg(
        url=""
    ),
    "mobilenet_v2_1.0_128": _cfg(
        url=""
    ),
    "mobilenet_v2_1.0_96": _cfg(
        url=""
    ),
    "mobilenet_v2_0.75_224": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/mobilenet/mobilenetv2/mobilenet_v2_075-243f9404.ckpt"
    ),
    "mobilenet_v2_0.75_192": _cfg(
        url=""
    ),
    "mobilenet_v2_0.75_160": _cfg(
        url=""
    ),
    "mobilenet_v2_0.75_128": _cfg(
        url=""
    ),
    "mobilenet_v2_0.75_96": _cfg(
        url=""
    ),
    "mobilenet_v2_0.5_224": _cfg(
        url=""
    ),
    "mobilenet_v2_0.5_192": _cfg(
        url=""
    ),
    "mobilenet_v2_0.5_160": _cfg(
        url=""
    ),
    "mobilenet_v2_0.5_128": _cfg(
        url=""
    ),
    "mobilenet_v2_0.5_96": _cfg(
        url=""
    ),
    "mobilenet_v2_0.35_224": _cfg(
        url=""
    ),
    "mobilenet_v2_0.35_192": _cfg(
        url=""
    ),
    "mobilenet_v2_0.35_160": _cfg(
        url=""
    ),
    "mobilenet_v2_0.35_128": _cfg(
        url=""
    ),
    "mobilenet_v2_0.35_96": _cfg(
        url=""
    ),
}


class InvertedResidual(nn.Cell):
    """Inverted Residual Block of MobileNetV2"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = round(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # pw
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, 1, pad_mode="pad", padding=0, has_bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6()
            ])
        layers.extend([
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad_mode="pad", padding=1, group=hidden_dim, has_bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, pad_mode="pad", padding=0, has_bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.layers = nn.SequentialCell(layers)

    def construct(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.layers(x)
        return self.layers(x)


class MobileNetV2(nn.Cell):
    r"""MobileNetV2 model class, based on
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_

    Args:
        alpha: scale factor of model width. Default: 1.
        round_nearest: divisor of make divisible function. Default: 8.
        in_channels: number the channels of the input. Default: 3.
        num_classes: number of classification classes. Default: 1000.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        round_nearest: int = 8,
        in_channels: int = 3,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        input_channels = make_divisible(32 * alpha, round_nearest)
        # Setting of inverted residual blocks.
        # t: The expansion factor.
        # c: Number of output channel.
        # n: Number of block.
        # s: First block stride.
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        last_channels = make_divisible(1280 * max(1.0, alpha), round_nearest)

        # Building stem conv layer.
        features = [
            nn.Conv2d(in_channels, input_channels, 3, 2, pad_mode="pad", padding=1, has_bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(),
        ]
        # Building inverted residual blocks.
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channels, output_channel, stride, expand_ratio=t))
                input_channels = output_channel
        # Building last point-wise layers.
        features.extend([
            nn.Conv2d(input_channels, last_channels, 1, 1, pad_mode="pad", padding=0, has_bias=False),
            nn.BatchNorm2d(last_channels),
            nn.ReLU6(),
        ])
        self.features = nn.SequentialCell(features)

        self.pool = GlobalAvgPooling()
        self.classifier = nn.SequentialCell([
            nn.Dropout(keep_prob=0.8),  # confirmed by paper authors
            nn.Dense(last_channels, num_classes),
        ])
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                n = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                cell.weight.set_data(
                    init.initializer(init.Normal(sigma=math.sqrt(2. / n), mean=0.0),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(init.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(init.initializer("zeros", cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(init.Normal(sigma=0.01, mean=0.0), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.features(x)
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
def mobilenet_v2_140_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model with width scaled by 1.4 and input image size of 224.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_1.4_224"]
    model = MobileNetV2(alpha=1.4, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_130_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model with width scaled by 1.3 and input image size of 224.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_1.3_224"]
    model = MobileNetV2(alpha=1.3, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_100_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model without width scaling and input image size of 224.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_1.0_224"]
    model = MobileNetV2(alpha=1.0, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_100_192(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model without width scaling and input image size of 192.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_1.0_192"]
    model = MobileNetV2(alpha=1.0, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_100_160(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model without width scaling and input image size of 160.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_1.0_160"]
    model = MobileNetV2(alpha=1.0, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_100_128(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model without width scaling and input image size of 128.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_1.0_128"]
    model = MobileNetV2(alpha=1.0, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_100_96(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model without width scaling and input image size of 96.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_1.0_96"]
    model = MobileNetV2(alpha=1.0, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_075_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model with width scaled by 0.75 and input image size of 224.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_0.75_224"]
    model = MobileNetV2(alpha=0.75, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_075_192(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model with width scaled by 0.75 and input image size of 192.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_0.75_192"]
    model = MobileNetV2(alpha=0.75, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_075_160(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model with width scaled by 0.75 and input image size of 160.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_0.75_160"]
    model = MobileNetV2(alpha=0.75, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_075_128(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model with width scaled by 0.75 and input image size of 128.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_0.75_128"]
    model = MobileNetV2(alpha=0.75, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_075_96(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model with width scaled by 0.75 and input image size of 96.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_0.75_96"]
    model = MobileNetV2(alpha=0.75, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_050_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model with width scaled by 0.5 and input image size of 224.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_0.5_224"]
    model = MobileNetV2(alpha=0.5, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_050_192(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model with width scaled by 0.5 and input image size of 192.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_0.5_192"]
    model = MobileNetV2(alpha=0.5, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_050_160(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model with width scaled by 0.5 and input image size of 160.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_0.5_160"]
    model = MobileNetV2(alpha=0.5, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_050_128(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model with width scaled by 0.5 and input image size of 128.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_0.5_128"]
    model = MobileNetV2(alpha=0.5, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_050_96(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model with width scaled by 0.5 and input image size of 96.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_0.5_96"]
    model = MobileNetV2(alpha=0.5, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_035_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model with width scaled by 0.35 and input image size of 224.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_0.35_224"]
    model = MobileNetV2(alpha=0.35, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_035_192(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model with width scaled by 0.35 and input image size of 192.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_0.35_192"]
    model = MobileNetV2(alpha=0.35, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_035_160(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model with width scaled by 0.35 and input image size of 160.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_0.35_160"]
    model = MobileNetV2(alpha=0.35, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_035_128(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model with width scaled by 0.35 and input image size of 128.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_0.35_128"]
    model = MobileNetV2(alpha=0.35, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_035_96(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    """Get MobileNetV2 model with width scaled by 0.35 and input image size of 96.
    Refer to the base class `models.MobileNetV2` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v2_0.35_96"]
    model = MobileNetV2(alpha=0.35, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
