from mindspore import nn

from ..utils.attention_cells import SEModule
from ._registry import register_backbone, register_backbone_class


class ConvBNLayer(nn.Cell):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 channels=None,
                 num_groups=1,
                 act=nn.HSwish,
                 has_bias=False):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            pad_mode='pad',
            padding=padding,
            group=num_groups,
            has_bias=has_bias)

        self.norm = nn.BatchNorm2d(num_filters)
        self.act = act()

    def construct(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class DepthwiseSeparable(nn.Cell):
    def __init__(self,
                 num_channels,
                 num_filters1,
                 num_filters2,
                 num_groups,
                 stride,
                 scale,
                 dw_size=3,
                 padding=1,
                 use_se=False):
        super(DepthwiseSeparable, self).__init__()
        self.use_se = use_se
        self.depthwise_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=int(num_filters1 * scale),
            filter_size=dw_size,
            stride=stride,
            padding=padding,
            num_groups=int(num_groups * scale))
        if use_se:
            self.se = SEModule(int(num_filters1 * scale))
        self.pointwise_conv = ConvBNLayer(
            num_channels=int(num_filters1 * scale),
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0)

    def construct(self, inputs):
        y = self.depthwise_conv(inputs)
        if self.use_se:
            y = self.se(y)
        y = self.pointwise_conv(y)
        return y


@register_backbone_class
class MobileNetV1Enhance(nn.Cell):
    def __init__(self,
                 in_channels=3,
                 scale=0.5,
                 last_conv_stride=[1, 2],
                 last_pool_type='max',
                 last_pool_kernel_size=[3, 2],
                 **kwargs):
        super().__init__()
        self.scale = scale
        self.block_list = []

        self.conv1 = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            channels=3,
            num_filters=int(32 * scale),
            stride=2,
            padding=1)

        conv2_1 = DepthwiseSeparable(
            num_channels=int(32 * scale),
            num_filters1=32,
            num_filters2=64,
            num_groups=32,
            stride=1,
            scale=scale)
        self.block_list.append(conv2_1)

        conv2_2 = DepthwiseSeparable(
            num_channels=int(64 * scale),
            num_filters1=64,
            num_filters2=128,
            num_groups=64,
            stride=1,
            scale=scale)
        self.block_list.append(conv2_2)

        conv3_1 = DepthwiseSeparable(
            num_channels=int(128 * scale),
            num_filters1=128,
            num_filters2=128,
            num_groups=128,
            stride=1,
            scale=scale)
        self.block_list.append(conv3_1)

        conv3_2 = DepthwiseSeparable(
            num_channels=int(128 * scale),
            num_filters1=128,
            num_filters2=256,
            num_groups=128,
            stride=(2, 1),
            scale=scale)
        self.block_list.append(conv3_2)

        conv4_1 = DepthwiseSeparable(
            num_channels=int(256 * scale),
            num_filters1=256,
            num_filters2=256,
            num_groups=256,
            stride=1,
            scale=scale)
        self.block_list.append(conv4_1)

        conv4_2 = DepthwiseSeparable(
            num_channels=int(256 * scale),
            num_filters1=256,
            num_filters2=512,
            num_groups=256,
            stride=(2, 1),
            scale=scale)
        self.block_list.append(conv4_2)

        for _ in range(5):
            conv5 = DepthwiseSeparable(
                num_channels=int(512 * scale),
                num_filters1=512,
                num_filters2=512,
                num_groups=512,
                stride=1,
                dw_size=5,
                padding=2,
                scale=scale,
                use_se=False)
            self.block_list.append(conv5)

        conv5_6 = DepthwiseSeparable(
            num_channels=int(512 * scale),
            num_filters1=512,
            num_filters2=1024,
            num_groups=512,
            stride=(2, 1),
            dw_size=5,
            padding=2,
            scale=scale,
            use_se=True)
        self.block_list.append(conv5_6)

        conv6 = DepthwiseSeparable(
            num_channels=int(1024 * scale),
            num_filters1=1024,
            num_filters2=1024,
            num_groups=1024,
            stride=tuple(last_conv_stride),
            dw_size=5,
            padding=2,
            use_se=True,
            scale=scale)
        self.block_list.append(conv6)

        self.block_list = nn.SequentialCell(*self.block_list)
        if last_pool_type == 'avg':
            self.pool = nn.AvgPool2d(
                kernel_size=tuple(last_pool_kernel_size),
                stride=tuple(last_pool_kernel_size),
                padding=0)
        else:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = [int(1024 * scale)]

    def construct(self, inputs):
        y = self.conv1(inputs)
        y = self.block_list(y)
        y = self.pool(y)
        return y


@register_backbone
def mobilenet_v1_enhance(scale=0.5,
                         last_conv_stride=[1, 2],
                         last_pool_type='avg',
                         last_pool_kernel_size=[2, 2],
                         pretrained=False,
                         **kwargs) -> MobileNetV1Enhance:
    model = MobileNetV1Enhance(scale=scale, last_conv_stride=last_conv_stride, last_pool_type=last_pool_type,
                               last_pool_kernel_size=last_pool_kernel_size, **kwargs)

    if pretrained is True:
        raise NotImplementedError

    return model
