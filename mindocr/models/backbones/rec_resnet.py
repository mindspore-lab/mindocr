from mindspore import nn

from ._registry import register_backbone, register_backbone_class

__all__ = ['RecResNet', 'rec_resnet34']


class ConvNormLayer(nn.Cell):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            is_vd_mode=False,
            act=False):
        super(ConvNormLayer, self).__init__()

        self.is_vd_mode = is_vd_mode
        self.pool2d_avg = nn.AvgPool2d(
            kernel_size=stride, stride=stride, pad_mode="same")
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1 if is_vd_mode else stride,
            pad_mode='pad',
            padding=(kernel_size - 1) // 2,
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.9,
                                   gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)
        self.act_func = nn.ReLU()
        self.act = act

    def construct(self, x):
        if self.is_vd_mode:
            x = self.pool2d_avg(x)
        y = self.conv(x)
        y = self.norm(y)
        if self.act:
            y = self.act_func(y)
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
        self.conv0 = ConvNormLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act=True)
        self.conv1 = ConvNormLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=False)

        if not shortcut:
            self.short = ConvNormLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                is_vd_mode=not if_first and stride[0] != 1)

        self.shortcut = shortcut
        self.relu = nn.ReLU()

    def construct(self, x):
        y = self.conv0(x)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)
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
        assert layers in supported_layers, "only support {} layers but input layer is {}".format(
            supported_layers, layers)

        depth = [3, 4, 6, 3]
        num_channels = [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        self.conv1_1 = ConvNormLayer(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act=True)
        self.conv1_2 = ConvNormLayer(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act=True)
        self.conv1_3 = ConvNormLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            act=True)
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        self.block_list = []
        for block_id in range(len(depth)):
            shortcut = False
            for i in range(depth[block_id]):
                if i == 0 and block_id != 0:
                    stride = (2, 1)
                else:
                    stride = (1, 1)

                is_first = block_id == i == 0
                in_channels = num_channels[block_id] if i == 0 else num_filters[block_id]
                basic_block = BasicBlock(
                    in_channels=in_channels,
                    out_channels=num_filters[block_id],
                    stride=stride,
                    shortcut=shortcut,
                    if_first=is_first
                )
                shortcut = True
                self.block_list.append(basic_block)

        self.block_list = nn.SequentialCell(self.block_list)
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same')

    def construct(self, x):
        y = self.conv1_1(x)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.maxpool2d_1(y)
        y = self.block_list(y)
        y = self.maxpool2d_2(y)
        return [y]


@register_backbone
def rec_resnet34(pretrained: bool = False, **kwargs):
    model = RecResNet(in_channels=3, layers=34, **kwargs)

    if pretrained is True:
        raise NotImplementedError("The default pretrained checkpoint for `rec_resnet34` backbone does not exist.")

    return model
