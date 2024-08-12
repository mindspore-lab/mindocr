import math
import mindspore as ms

from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from ._registry import register_backbone, register_backbone_class

ms.set_context(pynative_synchronize=True)

__all__ = ['DenseNet']

class Bottleneck(nn.Cell):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(
            nChannels,
            interChannels,
            kernel_size=1,
            has_bias=False,
            pad_mode='pad',
            padding=0,
        )
        self.bn2 = nn.BatchNorm2d(growthRate)
        self.conv2 = nn.Conv2d(
            interChannels,
            growthRate,
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
    def __init__(self, nChannels, growthRate, use_dropout):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(
            nChannels,
            growthRate,
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
    def __init__(self, nChannels, out_channels, use_dropout):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(
            nChannels,
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
    r'''The RecDenseNet model is the customized DenseNet backbone for Recognition.
    For example, in the CAN recognition algorithm, it is used in feature extraction to obtain a formula feature map.
    DenseNet Network is based on
    `"When Counting Meets HMER: Counting-Aware Network for Handwritten Mathematical Expression Recognition"
    <https://arxiv.org/abs/2207.11463>`_ paper.

    Args:
        growthRate (int): growth rate of DenseNet. The default value is 24.
        reduction (float): compression ratio in DenseNet. The default is 0.5.
        bottleneck (bool): specifies whether to use a bottleneck layer. The default is True.
        use_dropout (bool): indicates whether to use dropout. The default is True.
        input_channel (int): indicates the number of channels in the input image. The default is 3.
    Return:
        nn.Cell for backbone module

    Example:
        >>> # init a DenseNet network
        >>> params = {
                'growthRate': 24,
                'reduction': 0.5,
                'bottleneck': True,
                'use_dropout': True,
                'input_channel': 3,
            }
            model = DenseNet(**params)
    '''
    def __init__(self, growthRate, reduction, bottleneck, use_dropout, input_channel):
        super(DenseNet, self).__init__()
        nDenseBlocks = 16
        nChannels = 2 * growthRate

        self.conv1 = nn.Conv2d(
            input_channel,
            nChannels,
            kernel_size=7,
            stride=2,
            has_bias=False,
            pad_mode='pad',
            padding=3,
        )
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        out_channels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, out_channels, use_dropout)

        nChannels = out_channels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        out_channels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, out_channels, use_dropout)

        nChannels = out_channels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        self.out_channels = [684]

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
    
    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout):
        layers = []
        layer_constructor = Bottleneck if bottleneck else SingleLayer
        for _ in range(int(nDenseBlocks)):
            layers.append(layer_constructor(nChannels, growthRate, use_dropout))
            nChannels += growthRate
        return nn.SequentialCell(*layers)


@register_backbone
def rec_densenet(pretrained: bool = False, **kwargs) -> DenseNet:

    if pretrained is True:
        raise NotImplementedError("The default pretrained checkpoint for `rec_densenet` backbone does not exist.")

    model = DenseNet(**kwargs)
    return model
