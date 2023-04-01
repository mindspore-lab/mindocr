""" Selective Kernel Convolution/Attention
Paper: Selective Kernel Networks (https://arxiv.org/abs/1903.06586)
"""
from typing import List, Optional, Union

from mindspore import Tensor, nn, ops

from ..utils import make_divisible
from .conv_norm_act import Conv2dNormActivation
from .pooling import GlobalAvgPooling


def _kernel_valid(k):
    """Checks kernel size is valid or not."""
    if isinstance(k, (list, tuple)):
        for ki in k:
            # check if each element is valid, instead of early returning if the first element is valid.
            _kernel_valid(ki)
    else:
        assert k >= 3 and k % 2


class SelectiveKernelAttn(nn.Cell):
    """Selective Kernel Attention Module
    Selective Kernel attention mechanism factored out into its own module.
    """

    def __init__(
        self,
        channels: int,
        num_paths: int = 2,
        attn_channels: int = 32,
        activation: Optional[nn.Cell] = nn.ReLU,
        norm: Optional[nn.Cell] = nn.BatchNorm2d,
    ):
        super().__init__()
        self.num_paths = num_paths
        self.mean = GlobalAvgPooling(keep_dims=True)
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, has_bias=False)
        self.bn = norm(attn_channels)
        self.act = activation()
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1)
        self.softmax = nn.Softmax(axis=1)

    def construct(self, x: Tensor) -> Tensor:
        x = self.mean((x.sum(1)))
        x = self.fc_reduce(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc_select(x)
        b, c, h, w = x.shape
        x = x.reshape((b, self.num_paths, c // self.num_paths, h, w))
        x = self.softmax(x)
        return x


class SelectiveKernel(nn.Cell):
    """Selective Kernel Convolution Module
    As described in Selective Kernel Networks (https://arxiv.org/abs/1903.06586) with some modifications.
    Largest change is the input split, which divides the input channels across each convolution path, this can
    be viewed as a grouping of sorts, but the output channel counts expand to the module level value. This keeps
    the parameter count from ballooning when the convolutions themselves don't have groups, but still provides
    a noteworthy increase in performance over similar param count models without this attention layer. -Ross W
    Args:
        in_channels (int):  module input (feature) channel count
        out_channels (int):  module output (feature) channel count
        kernel_size (int, list): kernel size for each convolution branch
        stride (int): stride for convolutions
        dilation (int): dilation for module as a whole, impacts dilation of each branch
        groups (int): number of groups for each branch
        rd_ratio (int, float): reduction factor for attention features
        rd_channels(int): reduction channels can be specified directly by arg (if rd_channels is set)
        rd_divisor(int): divisor can be specified to keep channels
        keep_3x3 (bool): keep all branch convolution kernels as 3x3, changing larger kernels for dilations
        split_input (bool): split input channels evenly across each convolution branch, keeps param count lower,
            can be viewed as grouping by path, output expands to module out_channels count
        activation (nn.Module): activation layer to use
        norm (nn.Module): batchnorm/norm layer to use
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: Optional[Union[int, List]] = None,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        rd_ratio: float = 1.0 / 16,
        rd_channels: Optional[int] = None,
        rd_divisor: int = 8,
        keep_3x3: bool = True,
        split_input: bool = True,
        activation: Optional[nn.Cell] = nn.ReLU,
        norm: Optional[nn.Cell] = nn.BatchNorm2d,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        kernel_size = kernel_size or [3, 5]  # default to one 3x3 and one 5x5 branch. 5x5 -> 3x3 + dilation
        _kernel_valid(kernel_size)
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * 2
        if keep_3x3:
            dilation = [dilation * (k - 1) // 2 for k in kernel_size]
            kernel_size = [3] * len(kernel_size)
        else:
            dilation = [dilation] * len(kernel_size)
        self.num_paths = len(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_input = split_input
        if self.split_input:
            assert in_channels % self.num_paths == 0
            in_channels = in_channels // self.num_paths
        groups = min(out_channels, groups)

        self.paths = nn.CellList([
            Conv2dNormActivation(in_channels, out_channels, kernel_size=k, stride=stride, groups=groups,
                                 dilation=d, activation=activation, norm=norm)
            for k, d in zip(kernel_size, dilation)
        ])

        attn_channels = rd_channels or make_divisible(out_channels * rd_ratio, divisor=rd_divisor)
        self.attn = SelectiveKernelAttn(out_channels, self.num_paths, attn_channels)

    def construct(self, x: Tensor) -> Tensor:
        x_paths = []
        if self.split_input:
            x_split = ops.split(x, axis=1, output_num=self.num_paths)
            for i, op in enumerate(self.paths):
                x_paths.append(op(x_split[i]))
        else:
            for op in self.paths:
                x_paths.append(op(x))

        x = ops.stack(x_paths, axis=1)
        x_attn = self.attn(x)
        x = x * x_attn
        x = x.sum(1)
        return x
