"""
MindSpore implementation of pnasnet.
Refer to Progressive Neural Architecture Search.
"""

import math
from collections import OrderedDict

import mindspore.common.initializer as init
from mindspore import Tensor, nn, ops

from .layers import GlobalAvgPooling
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "Pnasnet",
    "pnasnet",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "conv_0.conv",
        "classifier": "last_linear",
        **kwargs,
    }


default_cfgs = {
    "pnasnet": _cfg(url=""),
}


class MaxPool(nn.Cell):
    """
    MaxPool: MaxPool2d with zero padding.
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int = 1,
        zero_pad: bool = False,
    ) -> None:
        super().__init__()
        self.pad = zero_pad
        if self.pad:
            self.zero_pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 0), (1, 0)))
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, pad_mode="same")

    def construct(self, x: Tensor) -> Tensor:
        if self.pad:
            x = self.zero_pad(x)
        x = self.pool(x)
        if self.pad:
            x = x[:, :, 1:, 1:]
        return x


class SeparableConv2d(nn.Cell):
    """
    SeparableConv2d: Separable convolutions consist of first performing
    a depthwise spatial convolution followed by a pointwise convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dw_kernel_size: int,
        dw_stride: int,
        dw_padding: int,
    ) -> None:
        super().__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                          kernel_size=dw_kernel_size, stride=dw_stride,
                                          pad_mode="pad", padding=dw_padding,
                                          group=in_channels, has_bias=False)
        self.pointwise_conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=1, pad_mode="pad", has_bias=False)

    def construct(self, x: Tensor) -> Tensor:
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Cell):
    """
    BranchSeparables: ReLU + Zero_Pad (when zero_pad is True) +  SeparableConv2d + BatchNorm2d +
                      ReLU + SeparableConv2d + BatchNorm2d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        stem_cell: bool = False,
        zero_pad: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        middle_channels = out_channels if stem_cell else in_channels

        self.pad = zero_pad
        if self.pad:
            self.zero_pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 0), (1, 0)))

        self.relu_1 = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, middle_channels,
                                           kernel_size, dw_stride=stride,
                                           dw_padding=padding)
        self.bn_sep_1 = nn.BatchNorm2d(num_features=middle_channels, eps=0.001, momentum=0.9)

        self.relu_2 = nn.ReLU()
        self.separable_2 = SeparableConv2d(middle_channels, out_channels,
                                           kernel_size, dw_stride=1,
                                           dw_padding=padding)
        self.bn_sep_2 = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.9)

    def construct(self, x: Tensor) -> Tensor:
        x = self.relu_1(x)
        if self.pad:
            x = self.zero_pad(x)
        x = self.separable_1(x)
        if self.pad:
            x = x[:, :, 1:, 1:]
        x = self.bn_sep_1(x)
        x = self.relu_2(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class ReluConvBn(nn.Cell):
    """
    ReluConvBn: ReLU + Conv2d + BatchNorm2d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.relu = nn.ReLU()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, pad_mode="pad", has_bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.9)

    def construct(self, x: Tensor) -> Tensor:
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class FactorizedReduction(nn.Cell):
    """
    FactorizedReduction is used to reduce the spatial size
    of the left input of a cell approximately by a factor of 2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.relu = nn.ReLU()

        path_1 = OrderedDict([
            ("avgpool", nn.AvgPool2d(kernel_size=1, stride=2, pad_mode="valid")),
            ("conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=1,
                               pad_mode="pad", has_bias=False)),
        ])
        self.path_1 = nn.SequentialCell(path_1)

        self.path_2 = nn.CellList([])
        self.path_2.append(nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 1)), mode="CONSTANT"))
        self.path_2.append(
            nn.AvgPool2d(kernel_size=1, stride=2, pad_mode="valid")
        )
        self.path_2.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2 + int(out_channels % 2),
                      kernel_size=1, stride=1, pad_mode="pad", has_bias=False)
        )

        self.final_path_bn = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.9)

    def construct(self, x: Tensor) -> Tensor:
        x = self.relu(x)
        x_path1 = self.path_1(x)

        x_path2 = self.path_2[0](x)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2[1](x_path2)
        x_path2 = self.path_2[2](x_path2)

        out = self.final_path_bn(ops.concat((x_path1, x_path2), axis=1))
        return out


class CellBase(nn.Cell):
    """
    CellBase: PNASNet base unit.
    """

    def cell_forward(self, x_left: Tensor, x_right: Tensor) -> Tensor:
        """
        cell_forward: to calculate the output according the x_left and x_right.
        """
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_left = self.comb_iter_3_left(x_comb_iter_2)
        x_comb_iter_3_right = self.comb_iter_3_right(x_right)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_left)
        if self.comb_iter_4_right is not None:
            x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        else:
            x_comb_iter_4_right = x_right
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = ops.concat((x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4), axis=1)

        return x_out


class CellStem0(CellBase):
    """
    CellStemp0:PNASNet Stem0 unit
    """

    def __init__(
        self,
        in_channels_left: int,
        out_channels_left: int,
        in_channels_right: int,
        out_channels_right: int,
    ) -> None:
        super().__init__()
        self.conv_1x1 = ReluConvBn(in_channels_right, out_channels_right,
                                   kernel_size=1)
        self.comb_iter_0_left = BranchSeparables(in_channels_left,
                                                 out_channels_left,
                                                 kernel_size=5, stride=2,
                                                 stem_cell=True)
        comb_iter_0_right = OrderedDict([
            ("max_pool", MaxPool(3, stride=2)),
            ("conv", nn.Conv2d(in_channels_left, out_channels_left,
                               kernel_size=1, has_bias=False)),
            ("bn", nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.9))
        ])
        self.comb_iter_0_right = nn.SequentialCell(comb_iter_0_right)

        self.comb_iter_1_left = BranchSeparables(out_channels_right,
                                                 out_channels_right,
                                                 kernel_size=7, stride=2)
        self.comb_iter_1_right = MaxPool(3, stride=2)
        self.comb_iter_2_left = BranchSeparables(out_channels_right,
                                                 out_channels_right,
                                                 kernel_size=5, stride=2)
        self.comb_iter_2_right = BranchSeparables(out_channels_right,
                                                  out_channels_right,
                                                  kernel_size=3, stride=2)
        self.comb_iter_3_left = BranchSeparables(out_channels_right,
                                                 out_channels_right,
                                                 kernel_size=3)
        self.comb_iter_3_right = MaxPool(3, stride=2)
        self.comb_iter_4_left = BranchSeparables(in_channels_right,
                                                 out_channels_right,
                                                 kernel_size=3, stride=2,
                                                 stem_cell=True)
        self.comb_iter_4_right = ReluConvBn(out_channels_right,
                                            out_channels_right,
                                            kernel_size=1, stride=2)

    def construct(self, x_left: Tensor) -> Tensor:
        x_right = self.conv_1x1(x_left)
        x_out = self.cell_forward(x_left, x_right)
        return x_out


class Cell(CellBase):
    """
    Cell class that is used as a 'layer' in image architectures
    """

    def __init__(
        self,
        in_channels_left: int,
        out_channels_left: int,
        in_channels_right: int,
        out_channels_right: int,
        is_reduction: bool = False,
        zero_pad: bool = False,
        match_prev_layer_dimensions: bool = False,
    ) -> None:
        super().__init__()

        stride = 2 if is_reduction else 1

        self.match_prev_layer_dimensions = match_prev_layer_dimensions
        if match_prev_layer_dimensions:
            self.conv_prev_1x1 = FactorizedReduction(in_channels_left, out_channels_left)
        else:
            self.conv_prev_1x1 = ReluConvBn(in_channels_left, out_channels_left, kernel_size=1)

        self.conv_1x1 = ReluConvBn(in_channels_right, out_channels_right, kernel_size=1)
        self.comb_iter_0_left = BranchSeparables(out_channels_left,
                                                 out_channels_left,
                                                 kernel_size=5, stride=stride,
                                                 zero_pad=zero_pad)
        self.comb_iter_0_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_1_left = BranchSeparables(out_channels_right,
                                                 out_channels_right,
                                                 kernel_size=7, stride=stride,
                                                 zero_pad=zero_pad)
        self.comb_iter_1_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_2_left = BranchSeparables(out_channels_right,
                                                 out_channels_right,
                                                 kernel_size=5, stride=stride,
                                                 zero_pad=zero_pad)
        self.comb_iter_2_right = BranchSeparables(out_channels_right,
                                                  out_channels_right,
                                                  kernel_size=3, stride=stride,
                                                  zero_pad=zero_pad)
        self.comb_iter_3_left = BranchSeparables(out_channels_right,
                                                 out_channels_right,
                                                 kernel_size=3)
        self.comb_iter_3_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_4_left = BranchSeparables(out_channels_left,
                                                 out_channels_left,
                                                 kernel_size=3, stride=stride,
                                                 zero_pad=zero_pad)
        if is_reduction:
            self.comb_iter_4_right = ReluConvBn(out_channels_right,
                                                out_channels_right,
                                                kernel_size=1, stride=stride)
        else:
            self.comb_iter_4_right = None

    def construct(self, x_left: Tensor, x_right: Tensor) -> Tensor:
        x_left = self.conv_prev_1x1(x_left)
        x_right = self.conv_1x1(x_right)
        x_out = self.cell_forward(x_left, x_right)
        return x_out


class Pnasnet(nn.Cell):
    r"""PNasNet model class, based on
    `"Progressive Neural Architecture Search" <https://arxiv.org/pdf/1712.00559.pdf>`_
    Args:
        number of input channels. Default: 3.
        num_classes: number of classification classes. Default: 1000.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.conv_0 = nn.SequentialCell(OrderedDict([
            ("conv", nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2,
                               pad_mode="pad", has_bias=False)),
            ("bn", nn.BatchNorm2d(num_features=32, eps=0.001, momentum=0.9))
        ]))

        self.cell_stem_0 = CellStem0(in_channels_left=32, out_channels_left=13,
                                     in_channels_right=32, out_channels_right=13)

        self.cell_stem_1 = Cell(in_channels_left=32, out_channels_left=27,
                                in_channels_right=65, out_channels_right=27,
                                match_prev_layer_dimensions=True,
                                is_reduction=True)
        self.cell_0 = Cell(in_channels_left=65, out_channels_left=54,
                           in_channels_right=135, out_channels_right=54,
                           match_prev_layer_dimensions=True)
        self.cell_1 = Cell(in_channels_left=135, out_channels_left=54,
                           in_channels_right=270, out_channels_right=54)
        self.cell_2 = Cell(in_channels_left=270, out_channels_left=54,
                           in_channels_right=270, out_channels_right=54)
        self.cell_3 = Cell(in_channels_left=270, out_channels_left=108,
                           in_channels_right=270, out_channels_right=108,
                           is_reduction=True, zero_pad=True)
        self.cell_4 = Cell(in_channels_left=270, out_channels_left=108,
                           in_channels_right=540, out_channels_right=108,
                           match_prev_layer_dimensions=True)

        self.cell_5 = Cell(in_channels_left=540, out_channels_left=108,
                           in_channels_right=540, out_channels_right=108)

        self.cell_6 = Cell(in_channels_left=540, out_channels_left=216,
                           in_channels_right=540, out_channels_right=216,
                           is_reduction=True)
        self.cell_7 = Cell(in_channels_left=540, out_channels_left=216,
                           in_channels_right=1080, out_channels_right=216,
                           match_prev_layer_dimensions=True)
        self.cell_8 = Cell(in_channels_left=1080, out_channels_left=216,
                           in_channels_right=1080, out_channels_right=216)

        self.relu = nn.ReLU()
        self.pool = GlobalAvgPooling()
        self.dropout = nn.Dropout(keep_prob=0.5)
        self.last_linear = nn.Dense(in_channels=1080, out_channels=num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for cells."""
        self.init_parameters_data()
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                n = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                cell.weight.set_data(
                    init.initializer(init.Normal(math.sqrt(2.0 / n), 0), cell.weight.shape, cell.weight.dtype)
                )
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(init.Zero(), cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(init.initializer(init.One(), cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(init.initializer(init.Zero(), cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(init.Normal(0.01, 0), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(init.Zero(), cell.bias.shape, cell.bias.dtype))

    def forward_features(self, x: Tensor) -> Tensor:
        x_conv_0 = self.conv_0(x)
        x_stem_0 = self.cell_stem_0(x_conv_0)
        x_stem_1 = self.cell_stem_1(x_conv_0, x_stem_0)

        x_cell_0 = self.cell_0(x_stem_0, x_stem_1)
        x_cell_1 = self.cell_1(x_stem_1, x_cell_0)
        x_cell_2 = self.cell_2(x_cell_0, x_cell_1)

        x_cell_3 = self.cell_3(x_cell_1, x_cell_2)
        x_cell_4 = self.cell_4(x_cell_2, x_cell_3)
        x_cell_5 = self.cell_5(x_cell_3, x_cell_4)

        x_cell_6 = self.cell_6(x_cell_4, x_cell_5)
        x_cell_7 = self.cell_7(x_cell_5, x_cell_6)
        x_cell_8 = self.cell_8(x_cell_6, x_cell_7)

        return x_cell_8

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def pnasnet(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> Pnasnet:
    """Get Pnasnet model.
    Refer to the base class `models.Pnasnet` for more details."""
    default_cfg = default_cfgs["pnasnet"]
    model = Pnasnet(in_channels=in_channels, num_classes=num_classes, **kwargs)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model
