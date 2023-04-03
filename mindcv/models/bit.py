"""
MindSpore implementation of `BiT_ResNet`.
Refer to Big Transfer (BiT): General Visual Representation Learning.
"""

from typing import List, Optional, Type, Union

import mindspore
from mindspore import Tensor, nn, ops

from .layers.pooling import GlobalAvgPooling
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "BiT_ResNet",
    "BiTresnet50",
    "BiTresnet50x3",
    "BiTresnet101",
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
    "BiTresnet50": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/bit/BiT_resnet50-1e4795a4.ckpt"),
    "BiTresnet50x3": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/bit/BiT_resnet50x3-a960f91f.ckpt"),
    "BiTresnet101": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/bit/BiT_resnet101-2efa9106.ckpt"),
}


class StdConv2d(nn.Conv2d):
    r"""Conv2d with Weight Standardization
    Args:
        in_channels(int): The channel number of the input tensor of the Conv2d layer.
        out_channels(int): The channel number of the output tensor of the Conv2d layer.
        kernel_size(int): Specifies the height and width of the 2D convolution kernel.
        stride(int): The movement stride of the 2D convolution kernel. Default: 1.
        pad_mode(str): Specifies padding mode. The optional values are "same", "valid", "pad". Default: "same".
        padding(int): The number of padding on the height and width directions of the input. Default: 0.
        group(int): Splits filter into groups. Default: 1.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        pad_mode="same",
        padding=0,
        group=1,
    ) -> None:
        super(StdConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad_mode,
            padding,
            group,
        )
        self.mean_op = ops.ReduceMean(keep_dims=True)

    def construct(self, x):
        w = self.weight
        m = self.mean_op(w, [1, 2, 3])
        v = w.var((1, 2, 3), keepdims=True)
        w = (w - m) / mindspore.ops.sqrt(v + 1e-10)
        output = self.conv2d(x, w)
        return output


class Bottleneck(nn.Cell):
    """define the basic block of BiT
    Args:
          in_channels(int): The channel number of the input tensor of the Conv2d layer.
          channels(int): The channel number of the output tensor of the middle Conv2d layer.
          stride(int): The movement stride of the 2D convolution kernel. Default: 1.
          groups(int): Number of groups for group conv in blocks. Default: 1.
          base_width(int): Base width of pre group hidden channel in blocks. Default: 64.
          norm(nn.Cell): Normalization layer in blocks. Default: None.
          down_sample(nn.Cell): Down sample in blocks. Default: None.
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        norm: Optional[nn.Cell] = None,
        down_sample: Optional[nn.Cell] = None,
    ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.GroupNorm

        width = int(channels * (base_width / 64.0)) * groups
        self.gn1 = norm(32, in_channels)
        self.conv1 = StdConv2d(in_channels, width, kernel_size=1, stride=1)
        self.gn2 = norm(32, width)
        self.conv2 = StdConv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, pad_mode="pad", group=groups)
        self.gn3 = norm(32, width)
        self.conv3 = StdConv2d(width, channels * self.expansion,
                               kernel_size=1, stride=1)

        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x: Tensor) -> Tensor:
        identity = x
        out = self.gn1(x)
        out = self.relu(out)

        residual = out

        out = self.conv1(out)

        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.gn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.down_sample is not None:
            identity = self.down_sample(residual)

        out += identity
        # out = self.relu(out)

        return out


class BiT_ResNet(nn.Cell):
    r"""BiT_ResNet model class, based on
    `"Big Transfer (BiT): General Visual Representation Learning" <https://arxiv.org/abs/1912.11370>`_
    Args:
        block(Union[Bottleneck]): block of BiT_ResNetv2.
        layers(tuple(int)): number of layers of each stage.
        wf(int): width of each layer. Default: 1.
        num_classes(int): number of classification classes. Default: 1000.
        in_channels(int): number the channels of the input. Default: 3.
        groups(int): number of groups for group conv in blocks. Default: 1.
        base_width(int): base width of pre group hidden channel in blocks. Default: 64.
        norm(nn.Cell): normalization layer in blocks. Default: None.
    """

    def __init__(
        self,
        block: Type[Union[Bottleneck]],
        layers: List[int],
        wf: int = 1,
        num_classes: int = 1000,
        in_channels: int = 3,
        groups: int = 1,
        base_width: int = 64,
        norm: Optional[nn.Cell] = None,
    ) -> None:
        super().__init__()

        if norm is None:
            norm = nn.GroupNorm

        self.norm: nn.Cell = norm  # add type hints to make pylint happy
        self.input_channels = 64 * wf
        self.groups = groups
        self.base_with = base_width

        self.conv1 = StdConv2d(in_channels, self.input_channels, kernel_size=7,
                               stride=2, pad_mode="pad", padding=3)
        self.pad = nn.ConstantPad2d(1, 0)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")

        self.layer1 = self._make_layer(block, 64 * wf, layers[0])
        self.layer2 = self._make_layer(block, 128 * wf, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256 * wf, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512 * wf, layers[3], stride=2)

        self.gn = norm(32, 2048 * wf)
        self.relu = nn.ReLU()
        self.pool = GlobalAvgPooling(keep_dims=True)
        self.classifier = nn.Conv2d(512 * block.expansion * wf, num_classes, kernel_size=1, has_bias=True)

    def _make_layer(
        self,
        block: Type[Union[Bottleneck]],
        channels: int,
        block_nums: int,
        stride: int = 1,
    ) -> nn.SequentialCell:
        """build model depending on cfgs"""
        down_sample = None

        if stride != 1 or self.input_channels != channels * block.expansion:
            down_sample = nn.SequentialCell([
                StdConv2d(self.input_channels, channels * block.expansion, kernel_size=1, stride=stride),
            ])

        layers = []
        layers.append(
            block(
                self.input_channels,
                channels,
                stride=stride,
                down_sample=down_sample,
                groups=self.groups,
                base_width=self.base_with,
                norm=self.norm,
            )
        )
        self.input_channels = channels * block.expansion

        for _ in range(1, block_nums):
            layers.append(
                block(
                    self.input_channels,
                    channels,
                    groups=self.groups,
                    base_width=self.base_with,
                    norm=self.norm,
                )
            )

        return nn.SequentialCell(layers)

    def root(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.pad(x)
        x = self.max_pool(x)
        return x

    def forward_features(self, x: Tensor) -> Tensor:
        """Network forward feature extraction."""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.gn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.root(x)
        x = self.forward_features(x)
        x = self.forward_head(x)
        assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[..., 0, 0]


@register_model
def BiTresnet50(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 50 layers ResNet model.
    Refer to the base class `models.BiT_Resnet` for more details.
    """
    default_cfg = default_cfgs["BiTresnet50"]
    model = BiT_ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def BiTresnet50x3(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 50 layers ResNet model.
     Refer to the base class `models.BiT_Resnet` for more details.
     """
    default_cfg = default_cfgs["BiTresnet50x3"]
    model = BiT_ResNet(Bottleneck, [3, 4, 6, 3], wf=3, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def BiTresnet101(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 101 layers ResNet model.
    Refer to the base class `models.BiT_Resnet` for more details.
    """
    default_cfg = default_cfgs["BiTresnet101"]
    model = BiT_ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
