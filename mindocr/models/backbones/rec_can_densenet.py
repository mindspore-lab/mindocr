"""
Rec_DenseNet model
"""
import math
import mindspore as ms

from mindspore import nn
from mindspore import ops
from ._registry import register_backbone, register_backbone_class

ms.set_context(pynative_synchronize=True)

__all__ = ['DenseNet']


class Bottleneck(nn.Cell):
    """Bottleneck block of rec_densenet"""
    def __init__(self, n_channels, growth_rate, use_dropout):
        super().__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv1 = nn.Conv2d(
            n_channels,
            inter_channels,
            kernel_size=1,
            has_bias=False,
            pad_mode='pad',
            padding=0,
        )
        self.bn2 = nn.BatchNorm2d(growth_rate)
        self.conv2 = nn.Conv2d(
            inter_channels,
            growth_rate,
            kernel_size=3,
            has_bias=False,
            pad_mode='pad',
            padding=1
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def construct(self, x):
        out = ops.relu(self.bn1(self.conv1(x)))
        if self.use_dropout:
            out = self.dropout(out)
        out = ops.relu(self.bn2(self.conv2(out)))
        if self.use_dropout:
            out = self.dropout(out)
        out = ops.concat((x, out), 1)
        return out


class SingleLayer(nn.Cell):
    """SingleLayer block of rec_densenet"""
    def __init__(self, n_channels, growth_rate, use_dropout):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(
            n_channels,
            growth_rate,
            kernel_size=3,
            has_bias=False,
            pad_mode='pad',
            padding=1
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def construct(self, x):
        out = self.conv1(ops.relu(x))
        if self.use_dropout:
            out = self.dropout(out)
        out = ops.concat((x, out), 1)
        return out


class Transition(nn.Cell):
    """Transition Module of rec_densenet"""
    def __init__(self, n_channels, out_channels, use_dropout):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(
            n_channels,
            out_channels,
            kernel_size=1,
            has_bias=False
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def construct(self, x):
        out = ops.relu(self.bn1(self.conv1(x)))

        if self.use_dropout:
            out = self.dropout(out)
        out = ops.avg_pool2d(out, 2, stride=2, ceil_mode=True)
        return out


@register_backbone_class
class DenseNet(nn.Cell):
    r"""The RecDenseNet model is the customized DenseNet backbone for
    Handwritten Mathematical Expression Recognition.
    For example, in the CAN recognition algorithm, it is used in
    feature extraction to obtain a formula feature map.
    DenseNet Network is based on
    `"When Counting Meets HMER: Counting-Aware Network for
    Handwritten Mathematical Expression Recognition"
    <https://arxiv.org/abs/2207.11463>`_ paper.

    Args:
        growth_rate (int): growth rate of DenseNet. The default value is 24.
        reduction (float): compression ratio in DenseNet. The default is 0.5.
        bottleneck (bool): specifies whether to use a bottleneck layer. The default is True.
        use_dropout (bool): indicates whether to use dropout. The default is True.
        input_channels (int): indicates the number of channels in the input image. The default is 3.
    Return:
        nn.Cell for backbone module

    Example:
        >>> # init a DenseNet network
        >>> params = {
        >>>     'growth_rate': 24,
        >>>     'reduction': 0.5,
        >>>     'bottleneck': True,
        >>>     'use_dropout': True,
        >>>     'input_channels': 3,
        >>> }
        >>> model = DenseNet(**params)
    """
    def __init__(self, growth_rate, reduction, bottleneck, use_dropout, input_channels):
        super().__init__()
        n_dense_blocks = 16
        n_channels = 2 * growth_rate

        self.conv1 = nn.Conv2d(
            input_channels,
            n_channels,
            kernel_size=7,
            stride=2,
            has_bias=False,
            pad_mode='pad',
            padding=3,
        )
        self.dense1 = self.make_dense(
            n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout
        )
        n_channels += n_dense_blocks * growth_rate
        out_channels = int(math.floor(n_channels * reduction))
        self.trans1 = Transition(n_channels, out_channels, use_dropout)

        n_channels = out_channels
        self.dense2 = self.make_dense(
            n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout
        )
        n_channels += n_dense_blocks * growth_rate
        out_channels = int(math.floor(n_channels * reduction))
        self.trans2 = Transition(n_channels, out_channels, use_dropout)

        n_channels = out_channels
        self.dense3 = self.make_dense(
            n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout
        )
        n_channels += n_dense_blocks * growth_rate
        self.out_channels = [n_channels]

    def construct(self, x):
        out = self.conv1(x)
        out = ops.relu(out)
        out = ops.max_pool2d(out, 2, ceil_mode=True)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        return out

    def make_dense(self, n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout):
        """Create dense_layer of DenseNet"""
        layers = []
        layer_constructor = Bottleneck if bottleneck else SingleLayer
        for _ in range(int(n_dense_blocks)):
            layers.append(layer_constructor(n_channels, growth_rate, use_dropout))
            n_channels += growth_rate
        return nn.SequentialCell(*layers)


@register_backbone
def rec_can_densenet(pretrained: bool = False, **kwargs) -> DenseNet:
    """Create a rec_densenet backbone model."""
    if pretrained is True:
        raise NotImplementedError(
            "The default pretrained checkpoint for `rec_densenet` backbone does not exist."
        )

    model = DenseNet(**kwargs)
    return model
