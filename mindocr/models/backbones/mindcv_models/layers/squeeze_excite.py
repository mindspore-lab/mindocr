""" Squeeze-and-Excitation Channel Attention
An SE implementation originally based on PyTorch SE-Net impl.
Has since evolved with additional functionality / configuration.
Paper: `Squeeze-and-Excitation Networks` - https://arxiv.org/abs/1709.01507
"""
from typing import Optional

from mindspore import Tensor, nn, ops

from ..utils import make_divisible
from .pooling import GlobalAvgPooling


class SqueezeExcite(nn.Cell):
    """SqueezeExcite Module as defined in original SE-Nets with a few additions.
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * customizable activation, normalization, and gate layer
    """

    def __init__(
        self,
        in_channels: int,
        rd_ratio: float = 1.0 / 16,
        rd_channels: Optional[int] = None,
        rd_divisor: int = 8,
        norm: Optional[nn.Cell] = None,
        act_layer: nn.Cell = nn.ReLU,
        gate_layer: nn.Cell = nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.act = act_layer()
        self.gate = gate_layer()
        if not rd_channels:
            rd_channels = make_divisible(in_channels * rd_ratio, rd_divisor)

        self.conv_reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=rd_channels,
            kernel_size=1,
            has_bias=True,
        )
        if self.norm:
            self.bn = nn.BatchNorm2d(rd_channels)
        self.conv_expand = nn.Conv2d(
            in_channels=rd_channels,
            out_channels=in_channels,
            kernel_size=1,
            has_bias=True,
        )
        self.pool = GlobalAvgPooling(keep_dims=True)

    def construct(self, x: Tensor) -> Tensor:
        x_se = self.pool(x)
        x_se = self.conv_reduce(x_se)
        if self.norm:
            x_se = self.bn(x_se)
        x_se = self.act(x_se)
        x_se = self.conv_expand(x_se)
        x_se = self.gate(x_se)
        x = x * x_se
        return x


class SqueezeExciteV2(nn.Cell):
    """SqueezeExcite Module as defined in original SE-Nets with a few additions.
    V1 uses 1x1conv to replace fc layers, and V2 uses nn.Dense to implement directly.
    """

    def __init__(
        self,
        in_channels: int,
        rd_ratio: float = 1.0 / 16,
        rd_channels: Optional[int] = None,
        rd_divisor: int = 8,
        norm: Optional[nn.Cell] = None,
        act_layer: nn.Cell = nn.ReLU,
        gate_layer: nn.Cell = nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.act = act_layer()
        self.gate = gate_layer()
        if not rd_channels:
            rd_channels = make_divisible(in_channels * rd_ratio, rd_divisor)

        self.conv_reduce = nn.Dense(
            in_channels=in_channels,
            out_channels=rd_channels,
            has_bias=True,
        )
        if self.norm:
            self.bn = nn.BatchNorm2d(rd_channels)
        self.conv_expand = nn.Dense(
            in_channels=rd_channels,
            out_channels=in_channels,
            has_bias=True,
        )
        self.pool = GlobalAvgPooling(keep_dims=False)

    def construct(self, x: Tensor) -> Tensor:
        x_se = self.pool(x)
        x_se = self.conv_reduce(x_se)
        if self.norm:
            x_se = self.bn(x_se)
        x_se = self.act(x_se)
        x_se = self.conv_expand(x_se)
        x_se = self.gate(x_se)
        x_se = ops.expand_dims(x_se, -1)
        x_se = ops.expand_dims(x_se, -1)
        x = x * x_se
        return x
