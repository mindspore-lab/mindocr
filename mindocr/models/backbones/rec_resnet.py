from mindspore import nn

from ._registry import register_backbone, register_backbone_class

__all__ = ['RecResNet', 'rec_resnet34', 'rec_resnet31']


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


class BasicBlockResNet31(nn.Cell):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, is_downsample=False):
        super().__init__()
        self.conv1 = self.conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = self.conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.is_downsample = is_downsample
        if is_downsample:
            self.downsample = nn.SequentialCell(
                nn.Conv2d(
                    in_channels,
                    channels * self.expansion,
                    1,
                    stride),
                nn.BatchNorm2d(channels * self.expansion))
        else:
            self.downsample = nn.SequentialCell()
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.is_downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def conv3x3(self, in_channel, out_channel, stride=1):
        return nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=3,
            stride=stride,
            pad_mode="pad",
            padding=1)


@register_backbone_class
class RecResNet(nn.Cell):
    r'''The RecResNet model is the customized ResNet backbone for Recognition.
        ResNet Network is based on
        `"Deep Residual Learning for Image Recognition"
        <https://arxiv.org/abs/1512.03385>`_ paper.

        Args:
            in_channels (int): the number of input channels of images. Default: 3.
            layers (int): the number of layers of ResNet, which defines the structure of ResNet. Default: 34.
            kwargs (dict): input args for the ResNet Network.

        Return:
            nn.Cell for backbone module

        Example:
            >>> # init a ResNet-34 network
            >>> model = RecResNet(in_channels=3, layers=34, **kwargs)
        '''

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


@register_backbone_class
class RecResNet31(nn.Cell):
    '''
    Args:
        in_channels (int): Number of channels of input image tensor.
        layers (list[int]): List of BasicBlock number for each stage.
        channels (list[int]): List of out_channels of Conv2d layer.
        last_stage_pool (bool): If True, add `MaxPool2d` layer to last stage.
    '''

    def __init__(self,
                 in_channels=3,
                 layers=[1, 2, 5, 3],
                 channels=[64, 128, 256, 256, 512, 512, 512],
                 last_stage_pool=False,
                 **kwargs):
        super(RecResNet31, self).__init__()
        assert isinstance(in_channels, int)
        assert isinstance(last_stage_pool, bool)

        self.out_channels = [channels[-1]]

        self.last_stage_pool = last_stage_pool

        # conv 1 (Conv Conv)
        self.conv1_1 = nn.Conv2d(
            in_channels, channels[0], kernel_size=3, stride=1, pad_mode="pad", padding=1)
        self.bn1_1 = nn.BatchNorm2d(channels[0])
        self.relu1_1 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(
            channels[0], channels[1], kernel_size=3, stride=1, pad_mode="pad", padding=1)
        self.bn1_2 = nn.BatchNorm2d(channels[1])
        self.relu1_2 = nn.ReLU()

        # conv 2 (Max-pooling, Residual block, Conv)
        self.pool2 = nn.MaxPool2d(
            kernel_size=2, stride=2)
        self.block2 = self._make_layer(channels[1], channels[2], layers[0])
        self.conv2 = nn.Conv2d(
            channels[2], channels[2], kernel_size=3, stride=1, pad_mode="pad", padding=1)
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.relu2 = nn.ReLU()

        # conv 3 (Max-pooling, Residual block, Conv)
        self.pool3 = nn.MaxPool2d(
            kernel_size=2, stride=2)
        self.block3 = self._make_layer(channels[2], channels[3], layers[1])
        self.conv3 = nn.Conv2d(
            channels[3], channels[3], kernel_size=3, stride=1, pad_mode="pad", padding=1)
        self.bn3 = nn.BatchNorm2d(channels[3])
        self.relu3 = nn.ReLU()

        # conv 4 (Max-pooling, Residual block, Conv)
        self.pool4 = nn.MaxPool2d(
            kernel_size=(2, 1), stride=(2, 1), pad_mode="same")
        self.block4 = self._make_layer(channels[3], channels[4], layers[2])
        self.conv4 = nn.Conv2d(
            channels[4], channels[4], kernel_size=3, stride=1, pad_mode="pad", padding=1)
        self.bn4 = nn.BatchNorm2d(channels[4])
        self.relu4 = nn.ReLU()

        # conv 5 ((Max-pooling), Residual block, Conv)
        self.pool5 = None
        if self.last_stage_pool:
            self.pool5 = nn.MaxPool2d(
                kernel_size=2, stride=2)
        self.block5 = self._make_layer(channels[4], channels[5], layers[3])
        self.conv5 = nn.Conv2d(
            channels[5], channels[5], kernel_size=3, stride=1, pad_mode="pad", padding=1)
        self.bn5 = nn.BatchNorm2d(channels[5])
        self.relu5 = nn.ReLU()

    def _make_layer(self, input_channels, output_channels, blocks):
        layers = []
        for _ in range(blocks):
            is_downsample = False
            if input_channels != output_channels:
                is_downsample = True

            layers.append(
                BasicBlockResNet31(
                    input_channels, output_channels, is_downsample=is_downsample))
            input_channels = output_channels

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)

        x = self.pool2(x)
        x = self.block2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.pool3(x)
        x = self.block3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.pool4(x)
        x = self.block4(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        if self.pool5 is not None:
            x = self.pool5(x)
        x = self.block5(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        return x


@register_backbone
def rec_resnet34(pretrained: bool = False, **kwargs):

    r'''A predefined ResNet-34 customized for Recognition.

    Args:
        pretrained (bool): whether to use the pretrained backbone. Default: False.

    Return:
        nn.Cell for ResNet backbone module

    .. note::
        The default pretrained checkpoint for `rec_resnet34` backbone is still under development.
    '''

    model = RecResNet(in_channels=3, layers=34, **kwargs)

    if pretrained is True:
        raise NotImplementedError("The default pretrained checkpoint for `rec_resnet34` backbone does not exist.")

    return model


@register_backbone
def rec_resnet31(pretrained: bool = False, **kwargs):
    model = RecResNet31(in_channels=3, **kwargs)

    if pretrained is True:
        raise NotImplementedError("The default pretrained checkpoint for `rec_resnet31` backbone does not exist.")

    return model
