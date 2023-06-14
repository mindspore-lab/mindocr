import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal

from ._registry import register_backbone, register_backbone_class

__all__ = ['RecVGG', 'rec_vgg7']


class Conv(nn.Cell):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, use_bn=False, pad_mode='pad', padding=0):
        super(Conv, self).__init__()

        self.Relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, pad_mode=pad_mode,
                              has_bias=True, bias_init=TruncatedNormal(), weight_init=TruncatedNormal(0.02))

        if use_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channel, eps=1e-4, momentum=0.9,
                                     gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)

        self.use_bn = use_bn

    def construct(self, x):
        out = self.conv(x)
        if self.use_bn:
            out = self.bn(out)
        out = self.Relu(out)
        return out


@register_backbone_class
class RecVGG(nn.Cell):
    """VGG Network structure"""

    def __init__(self, **kwargs):
        super().__init__()
        self.conv1 = Conv(3, 64, use_bn=False, padding=1)
        self.conv2 = Conv(64, 128, use_bn=False, padding=1)
        self.conv3 = Conv(128, 256, use_bn=True, padding=1)
        self.conv4 = Conv(256, 256, use_bn=False, padding=1)
        self.conv5 = Conv(256, 512, use_bn=True, padding=1)
        self.conv6 = Conv(512, 512, use_bn=False, padding=1)
        self.conv7 = Conv(512, 512, kernel_size=2,
                          pad_mode='valid', padding=0, use_bn=True)
        self.maxpool2d1 = nn.MaxPool2d(
            kernel_size=2, stride=2, pad_mode='same')
        self.maxpool2d2 = nn.MaxPool2d(kernel_size=(
            2, 1), stride=(2, 1), pad_mode='same')

        self.out_channels = 512

    def construct(self, x):
        x = self.conv1(x)
        x = self.maxpool2d1(x)
        x = self.conv2(x)
        x = self.maxpool2d1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2d2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool2d2(x)
        x = self.conv7(x)
        return [x]


@register_backbone
def rec_vgg7(pretrained: bool = False, **kwargs):
    model = RecVGG(**kwargs)

    if pretrained is True:
        raise NotImplementedError("The default pretrained checkpoint for `rec_vgg7` backbone does not exist.")

    return model
