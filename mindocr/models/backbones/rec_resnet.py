from typing import Tuple, List
from mindspore import nn, Tensor, ops
from ._registry import register_backbone, register_backbone_class

__all__ = ['RecResNet', 'rec_resnet34']


class ConvBNLayer(nn.Cell):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            groups=1,
            is_vd_mode=False,
            act=False):
        super(ConvBNLayer, self).__init__()

        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = nn.AvgPool2d(
            kernel_size=stride, stride=stride, pad_mode="same")
        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1 if is_vd_mode else stride,
            pad_mode='pad',
            padding=(kernel_size - 1) // 2,
            )
        self._batch_norm = nn.BatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.9,
                                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)
        self._act = nn.ReLU()
        self.act = act

    def construct(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self.act:
            y = self._act(y)
        return y


class BottleneckBlock(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 name=None):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act=True)
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act=True)
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=False)

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=stride,
                is_vd_mode=not if_first and stride[0] != 1)

        self.shortcut = shortcut
        self.relu = nn.ReLU()

    def construct(self, inputs):
        y = self.conv0(inputs)

        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = short + conv2
        y = self.relu(y)
        return y


class BasicBlock(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False
                 ):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act=True)
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=False)

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                is_vd_mode=not if_first and stride[0] != 1)

        self.shortcut = shortcut
        self.relu = nn.ReLU()

    def construct(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = short + conv1
        y = self.relu(y)
        return y


@register_backbone_class
class RecResNet(nn.Cell):
    def __init__(self, in_channels=3, layers=34, **kwargs):
        super(RecResNet, self).__init__()

        self.out_channels = 512
        self.layers = layers
        supported_layers = [34]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        self.conv1_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act=True)
        self.conv1_2 = ConvBNLayer(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act=True)
        self.conv1_3 = ConvBNLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            act=True)
        self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        self.block_list = []
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                if i == 0 and block != 0:
                    stride = (2, 1)
                else:
                    stride = (1, 1)

                basic_block = BasicBlock(
                                in_channels=num_channels[block]
                                if i == 0 else num_filters[block],
                                out_channels=num_filters[block],
                                stride=stride,
                                shortcut=shortcut,
                                if_first=block == i == 0
                                )
                shortcut = True
                self.block_list.append(basic_block)
            self.out_channels = num_filters[block]
        
        self.block_list = nn.SequentialCell(self.block_list)
        self.out_pool = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same')

    def construct(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.pool2d_max(y)
        y = self.block_list(y)
        y = self.out_pool(y)
        return [y]

# TODO: load pretrained weight in build_backbone or use a unify wrapper to load


@register_backbone
def rec_resnet34(pretrained: bool = True, **kwargs):
    model = RecResNet(in_channels=3, layers=34, **kwargs)

    # load pretrained weights
    if pretrained:
        raise NotImplementedError

    return model
