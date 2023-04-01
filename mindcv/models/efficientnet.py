"""EfficientNet Architecture."""

import copy
import math
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np

from mindspore import Tensor, nn
from mindspore.common import initializer as weight_init
from mindspore.common.initializer import Normal, Uniform

from .layers.activation import Swish
from .layers.drop_path import DropPath
from .layers.pooling import GlobalAvgPooling
from .layers.squeeze_excite import SqueezeExcite
from .registry import register_model
from .utils import load_pretrained, make_divisible

__all__ = [
    "EfficientNet",  # registration mechanism to use yaml configuration
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
    "efficientnet_v2_xl",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        # 'first_conv': 'features.0.features.0', 'classifier': 'classifier',
        **kwargs,
    }


default_cfgs = {
    "efficientnet_b0": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/efficientnet/efficientnet_b0-103ec70c.ckpt"),
    "efficientnet_b1": _cfg(url=""),
    "efficientnet_b2": _cfg(url=""),
    "efficientnet_b3": _cfg(url=""),
    "efficientnet_b4": _cfg(url=""),
    "efficientnet_b5": _cfg(url=""),
    "efficientnet_b6": _cfg(url=""),
    "efficientnet_b7": _cfg(url=""),
    "efficientnet_v2_s": _cfg(url=""),
    "efficientnet_v2_m": _cfg(url=""),
    "efficientnet_v2_l": _cfg(url=""),
    "efficientnet_v2_xl": _cfg(url=""),
}


class MBConvConfig:
    """
    The Parameters of MBConv which need to multiply the expand_ration.

    Args:
        expand_ratio (float): The Times of the num of out_channels with respect to in_channels.
        kernel_size (int): The kernel size of the depthwise conv.
        stride (int): The stride of the depthwise conv.
        in_chs (int): The input_channels of the MBConv Module.
        out_chs (int): The output_channels of the MBConv Module.
        num_layers (int): The num of MBConv Module.
        width_cnf: The ratio of the channel. Default: 1.0.
        depth_cnf: The ratio of num_layers. Default: 1.0.

    Returns:
        None

    Examples:
        >>> cnf = MBConvConfig(1, 3, 1, 32, 16, 1)
        >>> print(cnf.input_channels)
    """

    def __init__(
        self,
        expand_ratio: float,
        kernel_size: int,
        stride: int,
        in_chs: int,
        out_chs: int,
        num_layers: int,
        width_cnf: float = 1.0,
        depth_cnf: float = 1.0,
    ) -> None:
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.stride = stride
        self.input_channels = self.adjust_channels(in_chs, width_cnf)
        self.out_channels = self.adjust_channels(out_chs, width_cnf)
        self.num_layers = self.adjust_depth(num_layers, depth_cnf)

    @staticmethod
    def adjust_channels(channels: int, width_cnf: float, min_value: Optional[int] = None) -> int:
        """
        Calculate the width of MBConv.

        Args:
            channels (int): The number of channel.
            width_cnf (float): The ratio of channel.
            min_value (int, optional): The minimum number of channel. Default: None.

        Returns:
            int, the width of MBConv.
        """

        return make_divisible(channels * width_cnf, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_cnf: float) -> int:
        """
        Calculate the depth of MBConv.

        Args:
            num_layers (int): The number of MBConv Module.
            depth_cnf (float): The ratio of num_layers.

        Returns:
            int, the depth of MBConv.
        """

        return int(math.ceil(num_layers * depth_cnf))


class MBConv(nn.Cell):
    """
    MBConv Module.

    Args:
        cnf (MBConvConfig): The class which contains the parameters(in_channels, out_channels, nums_layers) and
            the functions which help calculate the parameters after multipling the expand_ratio.
        keep_prob: The dropout rate in MBConv. Default: 0.8.
        norm (nn.Cell): The BatchNorm Method. Default: None.
        se_layer (nn.Cell): The squeeze-excite Module. Default: SqueezeExcite.

    Returns:
        Tensor
    """

    def __init__(
        self,
        cnf: MBConvConfig,
        keep_prob: float = 0.8,
        norm: Optional[nn.Cell] = None,
        se_layer: Callable[..., nn.Cell] = SqueezeExcite,
    ) -> None:
        super().__init__()

        self.shortcut = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Cell] = []

        # expand conv: the out_channels is cnf.expand_ratio times of the in_channels.
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.extend([
                nn.Conv2d(cnf.input_channels, expanded_channels, kernel_size=1),
                norm(expanded_channels),
                Swish(),
            ])

        # depthwise conv: splits the filter into groups.
        layers.extend([
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=cnf.kernel_size,
                      stride=cnf.stride, group=expanded_channels),
            norm(expanded_channels),
            Swish(),
        ])

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(in_channels=expanded_channels, rd_channels=squeeze_channels, act_layer=Swish))

        # project
        layers.extend([
            nn.Conv2d(expanded_channels, cnf.out_channels, kernel_size=1),
            norm(cnf.out_channels),
        ])

        self.block = nn.SequentialCell(layers)
        self.dropout = DropPath(keep_prob)
        self.out_channels = cnf.out_channels

    def construct(self, x) -> Tensor:
        result = self.block(x)
        if self.shortcut:
            result = self.dropout(result)
            result += x
        return result


class FusedMBConvConfig(MBConvConfig):
    """FusedMBConvConfig"""

    # Stores information listed at Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel_size: int,
        stride: int,
        in_chs: int,
        out_chs: int,
        num_layers: int,
    ) -> None:
        super().__init__(expand_ratio, kernel_size, stride, in_chs, out_chs, num_layers)


class FusedMBConv(nn.Cell):
    """FusedMBConv"""

    def __init__(
        self,
        cnf: FusedMBConvConfig,
        keep_prob: float,
        norm: Optional[nn.Cell] = None,
    ) -> None:
        super().__init__()

        if not 1 <= cnf.stride <= 2:
            raise ValueError("illegal stride value")

        self.shortcut = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Cell] = []

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            # fused expand
            layers.extend([
                nn.Conv2d(cnf.input_channels, expanded_channels, kernel_size=cnf.kernel_size,
                          stride=cnf.stride),
                norm(expanded_channels),
                Swish(),
            ])

            # project
            layers.extend([
                nn.Conv2d(expanded_channels, cnf.out_channels, kernel_size=1),
                norm(cnf.out_channels),
            ])
        else:
            layers.extend([
                nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=cnf.kernel_size,
                          stride=cnf.stride),
                norm(cnf.out_channels),
                Swish(),
            ])

        self.block = nn.SequentialCell(layers)
        self.dropout = DropPath(keep_prob)
        self.out_channels = cnf.out_channels

    def construct(self, x) -> Tensor:
        result = self.block(x)
        if self.shortcut:
            result = self.dropout(result)
            result += x
        return result


class EfficientNet(nn.Cell):
    """
    EfficientNet architecture.
    `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Args:
        arch (str): The name of the model.
        dropout_rate (float): The dropout rate of efficientnet.
        width_mult (float): The ratio of the channel. Default: 1.0.
        depth_mult (float): The ratio of num_layers. Default: 1.0.
        in_channels (int): The input channels. Default: 3.
        num_classes (int): The number of class. Default: 1000.
        inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]], optional): The settings of block.
            Default: None.
        keep_prob (float): The dropout rate of MBConv. Default: 0.2.
        norm_layer (nn.Cell, optional): The normalization layer. Default: None.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, 1000)`.
    """

    def __init__(
        self,
        arch: str,
        dropout_rate: float,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        in_channels: int = 3,
        num_classes: int = 1000,
        inverted_residual_setting: Optional[Sequence[Union[MBConvConfig, FusedMBConvConfig]]] = None,
        keep_prob: float = 0.2,
        norm_layer: Optional[nn.Cell] = None,
    ) -> None:
        super().__init__()
        self.last_channel = None

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            if width_mult >= 1.6:
                norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.99)

        layers: List[nn.Cell] = []

        if not inverted_residual_setting:
            if arch.startswith("efficientnet_b"):
                bneck_conf = partial(MBConvConfig, width_cnf=width_mult, depth_cnf=depth_mult)
                inverted_residual_setting = [
                    bneck_conf(1, 3, 1, 32, 16, 1),
                    bneck_conf(6, 3, 2, 16, 24, 2),
                    bneck_conf(6, 5, 2, 24, 40, 2),
                    bneck_conf(6, 3, 2, 40, 80, 3),
                    bneck_conf(6, 5, 1, 80, 112, 3),
                    bneck_conf(6, 5, 2, 112, 192, 4),
                    bneck_conf(6, 3, 1, 192, 320, 1),
                ]
            elif arch.startswith("efficientnet_v2_s"):
                inverted_residual_setting = [
                    FusedMBConvConfig(1, 3, 1, 24, 24, 2),
                    FusedMBConvConfig(4, 3, 2, 24, 48, 4),
                    FusedMBConvConfig(4, 3, 2, 48, 64, 4),
                    MBConvConfig(4, 3, 2, 64, 128, 6),
                    MBConvConfig(6, 3, 1, 128, 160, 9),
                    MBConvConfig(6, 3, 2, 160, 256, 15),
                ]
                self.last_channel = 1280
            elif arch.startswith("efficientnet_v2_m"):
                inverted_residual_setting = [
                    FusedMBConvConfig(1, 3, 1, 24, 24, 3),
                    FusedMBConvConfig(4, 3, 2, 24, 48, 5),
                    FusedMBConvConfig(4, 3, 2, 48, 80, 5),
                    MBConvConfig(4, 3, 2, 80, 160, 7),
                    MBConvConfig(6, 3, 1, 160, 176, 14),
                    MBConvConfig(6, 3, 2, 176, 304, 18),
                    MBConvConfig(6, 3, 1, 304, 512, 5),
                ]
                self.last_channel = 1280
            elif arch.startswith("efficientnet_v2_l"):
                inverted_residual_setting = [
                    FusedMBConvConfig(1, 3, 1, 32, 32, 4),
                    FusedMBConvConfig(4, 3, 2, 32, 64, 7),
                    FusedMBConvConfig(4, 3, 2, 64, 96, 7),
                    MBConvConfig(4, 3, 2, 96, 192, 10),
                    MBConvConfig(6, 3, 1, 192, 224, 19),
                    MBConvConfig(6, 3, 2, 224, 384, 25),
                    MBConvConfig(6, 3, 1, 384, 640, 7),
                ]
                self.last_channel = 1280
            elif arch.startswith("efficientnet_v2_xl"):
                inverted_residual_setting = [
                    FusedMBConvConfig(1, 3, 1, 32, 32, 4),
                    FusedMBConvConfig(4, 3, 2, 32, 64, 8),
                    FusedMBConvConfig(4, 3, 2, 64, 96, 8),
                    MBConvConfig(4, 3, 2, 96, 192, 16),
                    MBConvConfig(6, 3, 1, 192, 256, 24),
                    MBConvConfig(6, 3, 2, 256, 512, 32),
                    MBConvConfig(6, 3, 1, 512, 640, 8),
                ]
                self.last_channel = 1280

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.extend([
            nn.Conv2d(in_channels, firstconv_output_channels, kernel_size=3, stride=2),
            norm_layer(firstconv_output_channels),
            Swish(),
        ])

        # building MBConv blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0

        # cnf is the settings of block
        for cnf in inverted_residual_setting:
            stage: List[nn.Cell] = []

            # cnf.num_layers is the num of the same block
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                block = MBConv

                if "FusedMBConvConfig" in str(type(block_cnf)):
                    block = FusedMBConv

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust dropout rate of blocks based on the depth of the stage block
                sd_prob = keep_prob * float(stage_block_id + 0.00001) / total_stage_blocks

                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.SequentialCell(stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = self.last_channel if self.last_channel is not None else 4 * lastconv_input_channels
        layers.extend([
            nn.Conv2d(lastconv_input_channels, lastconv_output_channels, kernel_size=1),
            norm_layer(lastconv_output_channels),
            Swish(),
        ])

        self.features = nn.SequentialCell(layers)
        self.avgpool = GlobalAvgPooling()
        self.dropout = nn.Dropout(1 - dropout_rate)
        self.mlp_head = nn.Dense(lastconv_output_channels, num_classes)
        self._initialize_weights()

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)

        if self.training:
            x = self.dropout(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        return self.mlp_head(x)

    def construct(self, x: Tensor) -> Tensor:
        """construct"""
        x = self.forward_features(x)
        return self.forward_head(x)

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                init_range = 1.0 / np.sqrt(cell.weight.shape[0])
                cell.weight.set_data(weight_init.initializer(Uniform(init_range), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(), cell.bias.shape, cell.bias.dtype))
            if isinstance(cell, nn.Conv2d):
                out_channel, _, kernel_size_h, kernel_size_w = cell.weight.shape
                stddev = np.sqrt(2 / int(out_channel * kernel_size_h * kernel_size_w))
                cell.weight.set_data(
                    weight_init.initializer(Normal(sigma=stddev), cell.weight.shape, cell.weight.dtype)
                )
                if cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(), cell.bias.shape, cell.bias.dtype))


def _efficientnet(
    arch: str,
    width_mult: float,
    depth_mult: float,
    dropout: float,
    in_channels: int,
    num_classes: int,
    pretrained: bool,
    **kwargs: Any,
) -> EfficientNet:
    """EfficientNet architecture."""

    model = EfficientNet(arch, dropout, width_mult, depth_mult, in_channels, num_classes, **kwargs)
    default_cfg = default_cfgs[arch]

    if pretrained:
        # Download the pretrained checkpoint file from url, and load
        # checkpoint file.
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def efficientnet_b0(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> EfficientNet:
    """
    Constructs a EfficientNet B0 architecture from
    `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pretrained on IMAGENET. Default: False.
        num_classes (int): The numbers of classes. Default: 1000.
        in_channels (int): The input channels. Default: 1000.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.
    """
    return _efficientnet("efficientnet_b0", 1.0, 1.0, 0.2, in_channels, num_classes, pretrained, **kwargs)


@register_model
def efficientnet_b1(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> EfficientNet:
    """
    Constructs a EfficientNet B1 architecture from
    `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pretrained on IMAGENET. Default: False.
        num_classes (int): The numbers of classes. Default: 1000.
        in_channels (int): The input channels. Default: 1000.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.
    """
    return _efficientnet("efficientnet_b1", 1.0, 1.1, 0.2, in_channels, num_classes, pretrained, **kwargs)


@register_model
def efficientnet_b2(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> EfficientNet:
    """
    Constructs a EfficientNet B2 architecture from
    `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pretrained on IMAGENET. Default: False.
        num_classes (int): The numbers of classes. Default: 1000.
        in_channels (int): The input channels. Default: 1000.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.
    """
    return _efficientnet("efficientnet_b2", 1.1, 1.2, 0.3, in_channels, num_classes, pretrained, **kwargs)


@register_model
def efficientnet_b3(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> EfficientNet:
    """
    Constructs a EfficientNet B3 architecture from
    `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pretrained on IMAGENET. Default: False.
        num_classes (int): The numbers of classes. Default: 1000.
        in_channels (int): The input channels. Default: 1000.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.
    """
    return _efficientnet("efficientnet_b3", 1.2, 1.4, 0.3, in_channels, num_classes, pretrained, **kwargs)


@register_model
def efficientnet_b4(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> EfficientNet:
    """
    Constructs a EfficientNet B4 architecture from
    `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pretrained on IMAGENET. Default: False.
        num_classes (int): The numbers of classes. Default: 1000.
        in_channels (int): The input channels. Default: 1000.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.
    """
    return _efficientnet("efficientnet_b4", 1.4, 1.8, 0.4, in_channels, num_classes, pretrained, **kwargs)


@register_model
def efficientnet_b5(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> EfficientNet:
    """
    Constructs a EfficientNet B5 architecture from
    `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pretrained on IMAGENET. Default: False.
        num_classes (int): The numbers of classes. Default: 1000.
        in_channels (int): The input channels. Default: 1000.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.
    """
    return _efficientnet("efficientnet_b5", 1.6, 2.2, 0.4, in_channels, num_classes, pretrained, **kwargs)


@register_model
def efficientnet_b6(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> EfficientNet:
    """
    Constructs a EfficientNet B6 architecture from
    `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pretrained on IMAGENET. Default: False.
        num_classes (int): The numbers of classes. Default: 1000.
        in_channels (int): The input channels. Default: 1000.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.
    """
    return _efficientnet("efficientnet_b6", 1.8, 2.6, 0.5, in_channels, num_classes, pretrained, **kwargs)


@register_model
def efficientnet_b7(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> EfficientNet:
    """
    Constructs a EfficientNet B7 architecture from
    `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pretrained on IMAGENET. Default: False.
        num_classes (int): The numbers of classes. Default: 1000.
        in_channels (int): The input channels. Default: 1000.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.
    """
    return _efficientnet("efficientnet_b7", 2.0, 3.1, 0.5, in_channels, num_classes, pretrained, **kwargs)


@register_model
def efficientnet_v2_s(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> EfficientNet:
    """
    Constructs a EfficientNet B4 architecture from
    `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pretrained on IMAGENET. Default: False.
        num_classes (int): The numbers of classes. Default: 1000.
        in_channels (int): The input channels. Default: 1000.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.
    """
    return _efficientnet("efficientnet_v2_s", 1.0, 1.0, 0.2, in_channels, num_classes, pretrained, **kwargs)


@register_model
def efficientnet_v2_m(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> EfficientNet:
    """
    Constructs a EfficientNet B4 architecture from
    `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pretrained on IMAGENET. Default: False.
        num_classes (int): The numbers of classes. Default: 1000.
        in_channels (int): The input channels. Default: 1000.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.
    """
    return _efficientnet("efficientnet_v2_m", 1.0, 1.0, 0.2, in_channels, num_classes, pretrained, **kwargs)


@register_model
def efficientnet_v2_l(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> EfficientNet:
    """
    Constructs a EfficientNet B4 architecture from
    `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pretrained on IMAGENET. Default: False.
        num_classes (int): The numbers of classes. Default: 1000.
        in_channels (int): The input channels. Default: 1000.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.
    """
    return _efficientnet("efficientnet_v2_l", 1.0, 1.0, 0.2, in_channels, num_classes, pretrained, **kwargs)


@register_model
def efficientnet_v2_xl(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> EfficientNet:
    """
    Constructs a EfficientNet B4 architecture from
    `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pretrained on IMAGENET. Default: False.
        num_classes (int): The numbers of classes. Default: 1000.
        in_channels (int): The input channels. Default: 1000.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.
    """
    return _efficientnet("efficientnet_v2_xl", 1.0, 1.0, 0.2, in_channels, num_classes, pretrained, **kwargs)
