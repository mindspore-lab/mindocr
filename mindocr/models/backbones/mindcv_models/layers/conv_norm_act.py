""" Conv2d + BN + Act"""
from typing import Optional

from mindspore import nn


class Conv2dNormActivation(nn.Cell):
    """Conv2d + BN + Act"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        pad_mode: str = "pad",
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        norm: Optional[nn.Cell] = nn.BatchNorm2d,
        activation: Optional[nn.Cell] = nn.ReLU,
        has_bias: Optional[bool] = None,
        **kwargs
    ) -> None:
        super().__init__()

        if pad_mode == "pad":
            if padding is None:
                padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        else:
            padding = 0

        if has_bias is None:
            has_bias = norm is None

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                pad_mode=pad_mode,
                padding=padding,
                dilation=dilation,
                group=groups,
                has_bias=has_bias,
                **kwargs
            )
        ]

        if norm:
            layers.append(norm(out_channels))
        if activation:
            layers.append(activation())

        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output
