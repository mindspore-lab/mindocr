"""
MindSpore implementation of `SENet`.
Refer to Squeeze-and-Excitation Networks.
"""

import math
from typing import List, Optional, Type, Union

import mindspore.common.initializer as init
from mindspore import Tensor, nn

from .layers.pooling import GlobalAvgPooling
from .layers.squeeze_excite import SqueezeExciteV2
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "SENet",
    "senet154",
    "seresnet18",
    "seresnet34",
    "seresnet50",
    "seresnet101",
    "seresnet152",
    "seresnext26_32x4d",
    "seresnext50_32x4d",
    "seresnext101_32x4d",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "layer0.0",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "senet154": _cfg(url=""),
    "seresnet18": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/senet/seresnet18-7880643b.ckpt"),
    "seresnet34": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/senet/seresnet34-8179d3c9.ckpt"),
    "seresnet50": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/senet/seresnet50-ff9cd214.ckpt"),
    "seresnet101": _cfg(url=""),
    "seresnet152": _cfg(url=""),
    "seresnext26_32x4d": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/senet/seresnext26_32x4d-5361f5b6.ckpt"
    ),
    "seresnext50_32x4d": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/senet/seresnext50_32x4d-fdc35aca.ckpt"
    ),
    "seresnext101_32x4d": _cfg(url=""),
}


class Bottleneck(nn.Cell):
    """
    Define the base block class for [SEnet, SEResNet, SEResNext] bottlenecks
    that implements `construct` method.
    """

    def construct(self, x: Tensor) -> Tensor:
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = self.se_module(out) + shortcut
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Define the Bottleneck for SENet154.
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        group: int,
        reduction: int,
        stride: int = 1,
        downsample: Optional[nn.SequentialCell] = None,
    ) -> None:
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels * 2, kernel_size=1, pad_mode="pad",
                               padding=0, has_bias=False)
        self.bn1 = nn.BatchNorm2d(channels * 2)
        self.conv2 = nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=stride,
                               pad_mode="pad", padding=1, group=group, has_bias=False)
        self.bn2 = nn.BatchNorm2d(channels * 4)
        self.conv3 = nn.Conv2d(channels * 4, channels * 4, kernel_size=1, pad_mode="pad",
                               padding=0, has_bias=False)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.relu = nn.ReLU()
        self.se_module = SqueezeExciteV2(channels * 4, rd_ratio=1.0 / reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    Define the ResNet bottleneck with a Squeeze-and-Excitation module,
    and the latter is used in the torchvision implementation of ResNet.
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        group: int,
        reduction: int,
        stride: int = 1,
        downsample: Optional[nn.SequentialCell] = None,
    ) -> None:
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, pad_mode="pad",
                               padding=0, has_bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, pad_mode="pad",
                               padding=1, group=group, has_bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * 4, kernel_size=1, pad_mode="pad", padding=0,
                               has_bias=False)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.relu = nn.ReLU()
        self.se_module = SqueezeExciteV2(channels * 4, rd_ratio=1.0 / reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    Define the ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        group: int,
        reduction: int,
        stride: int = 1,
        downsample: Optional[nn.SequentialCell] = None,
        base_width: int = 4,
    ) -> None:
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(channels * (base_width / 64)) * group
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, pad_mode="pad",
                               padding=0, has_bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, pad_mode="pad",
                               padding=1, group=group, has_bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, channels * 4, kernel_size=1, pad_mode="pad", padding=0,
                               has_bias=False)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.relu = nn.ReLU()
        self.se_module = SqueezeExciteV2(channels * 4, rd_ratio=1.0 / reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBlock(nn.Cell):
    """
    Define the basic block of resnet with a Squeeze-and-Excitation module.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        group: int,
        reduction: int,
        stride: int = 1,
        downsample: Optional[nn.SequentialCell] = None,
    ) -> None:
        super(SEResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, pad_mode="pad",
                               padding=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, pad_mode="pad", padding=1,
                               group=group, has_bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.se_module = SqueezeExciteV2(channels, rd_ratio=1.0 / reduction)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x: Tensor) -> Tensor:
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = self.se_module(out) + shortcut
        out = self.relu(out)

        return out


class SENet(nn.Cell):
    r"""SENet model class, based on
    `"Squeeze-and-Excitation Networks" <https://arxiv.org/abs/1709.01507>`_

    Args:
        block: block class of SENet.
        layers: Number of residual blocks for 4 layers.
        group: Number of groups for the conv in each bottleneck block.
        reduction: Reduction ratio for Squeeze-and-Excitation modules.
        drop_rate: Drop probability for the Dropout layer. Default: 0.
        in_channels: number the channels of the input. Default: 3.
        inplanes:  Number of input channels for layer1. Default: 64.
        input3x3: If `True`, use three 3x3 convolutions in layer0. Default: False.
        downsample_kernel_size: Kernel size for downsampling convolutions. Default: 1.
        downsample_padding: Padding for downsampling convolutions. Default: 0.
        num_classes (int): number of classification classes. Default: 1000.
    """

    def __init__(
        self,
        block: Type[Union[SEBottleneck, SEResNetBottleneck, SEResNetBlock, SEResNeXtBottleneck]],
        layers: List[int],
        group: int,
        reduction: int,
        drop_rate: float = 0.0,
        in_channels: int = 3,
        inplanes: int = 64,
        input3x3: bool = False,
        downsample_kernel_size: int = 1,
        downsample_padding: int = 0,
        num_classes: int = 1000,
    ) -> None:
        super(SENet, self).__init__()
        self.inplanes = inplanes
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        if input3x3:
            self.layer0 = nn.SequentialCell([
                nn.Conv2d(in_channels, 64, 3, stride=2, pad_mode="pad", padding=1, has_bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, pad_mode="pad", padding=1, has_bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, inplanes, 3, stride=1, pad_mode="pad", padding=1, has_bias=False),
                nn.BatchNorm2d(inplanes),
                nn.ReLU()
            ])
        else:
            self.layer0 = nn.SequentialCell([
                nn.Conv2d(in_channels, inplanes, kernel_size=7, stride=2, pad_mode="pad",
                          padding=3, has_bias=False),
                nn.BatchNorm2d(inplanes),
                nn.ReLU()
            ])
        self.pool0 = nn.MaxPool2d(3, stride=2, pad_mode="same")

        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0], group=group,
                                       reduction=reduction, downsample_kernel_size=1,
                                       downsample_padding=0)

        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1], stride=2,
                                       group=group, reduction=reduction,
                                       downsample_kernel_size=downsample_kernel_size,
                                       downsample_padding=downsample_padding)

        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2], stride=2,
                                       group=group, reduction=reduction,
                                       downsample_kernel_size=downsample_kernel_size,
                                       downsample_padding=downsample_padding)

        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3], stride=2,
                                       group=group, reduction=reduction,
                                       downsample_kernel_size=downsample_kernel_size,
                                       downsample_padding=downsample_padding)

        self.num_features = 512 * block.expansion

        self.pool = GlobalAvgPooling()
        if self.drop_rate > 0.:
            self.dropout = nn.Dropout(keep_prob=1. - self.drop_rate)
        self.classifier = nn.Dense(self.num_features, self.num_classes)

        self._initialize_weights()

    def _make_layer(
        self,
        block: Type[Union[SEBottleneck, SEResNetBottleneck, SEResNetBlock, SEResNeXtBottleneck]],
        planes: int,
        blocks: int,
        group: int,
        reduction: int,
        stride: int = 1,
        downsample_kernel_size: int = 1,
        downsample_padding: int = 0,
    ) -> nn.SequentialCell:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=downsample_kernel_size,
                          stride=stride, pad_mode="pad", padding=downsample_padding, has_bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            ])

        layers = [block(self.inplanes, planes, group, reduction, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, group, reduction))

        return nn.SequentialCell(layers)

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    init.initializer(init.HeNormal(mode="fan_out", nonlinearity="relu"),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer("zeros", cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(init.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(init.initializer("zeros", cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(init.HeUniform(mode="fan_in", nonlinearity="sigmoid"),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.layer0(x)
        x = self.pool0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        if self.drop_rate > 0.0:
            x = self.dropout(x)
        x = self.classifier(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def senet154(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["senet154"]
    model = SENet(block=SEBottleneck, layers=[3, 8, 36, 3], group=64, reduction=16,
                  downsample_kernel_size=3, downsample_padding=1,  inplanes=128, input3x3=True,
                  num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def seresnet18(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["seresnet18"]
    model = SENet(block=SEResNetBlock, layers=[2, 2, 2, 2], group=1, reduction=16,
                  num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def seresnet34(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["seresnet34"]
    model = SENet(block=SEResNetBlock, layers=[3, 4, 6, 3], group=1, reduction=16,
                  num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def seresnet50(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["seresnet50"]
    model = SENet(block=SEResNetBottleneck, layers=[3, 4, 6, 3], group=1, reduction=16,
                  num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def seresnet101(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["seresnet101"]
    model = SENet(block=SEResNetBottleneck, layers=[3, 4, 23, 3], group=1, reduction=16,
                  num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def seresnet152(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["seresnet152"]
    model = SENet(block=SEResNetBottleneck, layers=[3, 8, 36, 3], group=1, reduction=16,
                  num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def seresnext26_32x4d(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["seresnext26_32x4d"]
    model = SENet(block=SEResNeXtBottleneck, layers=[2, 2, 2, 2], group=32, reduction=16,
                  num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def seresnext50_32x4d(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["seresnext50_32x4d"]
    model = SENet(block=SEResNeXtBottleneck, layers=[3, 4, 6, 3], group=32, reduction=16,
                  num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def seresnext101_32x4d(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["seresnext101_32x4d"]
    model = SENet(block=SEResNeXtBottleneck, layers=[3, 4, 23, 3], group=32, reduction=16,
                  num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
