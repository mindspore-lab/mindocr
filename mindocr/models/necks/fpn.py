from typing import List, Tuple

from mindspore import Tensor, nn, ops
from mindspore.common.initializer import TruncatedNormal, XavierUniform

from .asf import AdaptiveScaleFusion


def _resize_nn(x: Tensor, scale: int = 0, shape: Tuple[int] = None):
    if scale == 1 or shape == x.shape[2:]:
        return x

    if scale:
        shape = (x.shape[2] * scale, x.shape[3] * scale)
    return ops.ResizeNearestNeighbor(shape)(x)


class FPN(nn.Cell):
    def __init__(self, in_channels, out_channels=256, **kwargs):
        super().__init__()
        self.out_channels = out_channels

    def construct(self, x):
        x1, x2, x3, x4 = x
        return x1


class DBFPN(nn.Cell):
    """
    DBNet & DBNet++'s Feature Pyramid Network that combines features extracted from a backbone at different scales
    into a single feature map.

    Args:
        in_channels: list of channel numbers of each input feature (e.g. resnet18 is [64, 128, 256, 512] and
                     resnet50 is [2048,1024,512,256]).
        out_channels: number of channels in the combined output feature map. Default: 256.
        weight_init: weights initialization method. Default: 'HeUniform'.
        bias: use bias in Conv2d operations. Default: False.
        use_asf: use Adaptive Scale Fusion module for multiscaled feature aggregation (DBNet++ only).
        channel_attention: use channel attention in addition to spatial and scale attentions in ASF module.
                           Default: True.
    """

    def __init__(self, in_channels: list, out_channels: int = 256, weight_init: str = 'HeUniform',
                 bias: bool = False, use_asf: bool = False, channel_attention: bool = True):
        super().__init__()
        self.out_channels = out_channels

        self.unify_channels = nn.CellList(
            [nn.Conv2d(ch, out_channels, 1, pad_mode='valid', has_bias=bias, weight_init=weight_init)
             for ch in in_channels]
        )

        self.out = nn.CellList(
            [nn.Conv2d(out_channels, out_channels // 4, 3, padding=1, pad_mode='pad', has_bias=bias,
                       weight_init=weight_init) for _ in range(len(in_channels))]
        )

        self.fuse = AdaptiveScaleFusion(out_channels, channel_attention, weight_init) if use_asf else ops.Concat(axis=1)

    def construct(self, features: List[Tensor]) -> Tensor:
        for i, uc_op in enumerate(self.unify_channels):
            features[i] = uc_op(features[i])

        for i in range(2, -1, -1):
            features[i] += _resize_nn(features[i + 1], shape=features[i].shape[2:])

        for i, out in enumerate(self.out):
            features[i] = _resize_nn(out(features[i]), shape=features[0].shape[2:])

        return self.fuse(features[::-1])  # matching the reverse order of the original work


def _conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='same', has_bias=False):
    init_value = TruncatedNormal(0.02)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     pad_mode=pad_mode, weight_init=init_value, has_bias=has_bias)


def _bn(channels, momentum=0.1):
    return nn.BatchNorm2d(channels, momentum=momentum)


class FCEFPN(nn.Cell):

    def __init__(self, in_channels, out_channel):
        in_channels = in_channels[1:]
        super(FCEFPN, self).__init__()

        self.reduce_conv_c3 = self.Xavier_conv(in_channels[0], out_channel, kernel_size=1, has_bias=True)

        self.reduce_conv_c4 = self.Xavier_conv(in_channels[1], out_channel, kernel_size=1, has_bias=True)

        self.reduce_conv_c5 = self.Xavier_conv(in_channels[2], out_channel, kernel_size=1, has_bias=True)

        self.smooth_conv_p5 = self.Xavier_conv(out_channel, out_channel, kernel_size=3, padding=1, pad_mode='pad',
                                               has_bias=True)

        self.smooth_conv_p4 = self.Xavier_conv(out_channel, out_channel, kernel_size=3, padding=1, pad_mode='pad',
                                               has_bias=True)

        self.smooth_conv_p3 = self.Xavier_conv(out_channel, out_channel, kernel_size=3, padding=1, pad_mode='pad',
                                               has_bias=True)

        self.out_channels = out_channel

    def construct(self, features):
        _, c3, c4, c5 = features

        p5 = self.reduce_conv_c5(c5)

        c4 = self.reduce_conv_c4(c4)
        p4 = ops.interpolate(p5, scale_factor=(2.0, 2.0), mode="area") + c4
        c3 = self.reduce_conv_c3(c3)
        p3 = ops.interpolate(p4, scale_factor=(2.0, 2.0), mode="area") + c3

        p5 = self.smooth_conv_p5(p5)
        p4 = self.smooth_conv_p4(p4)
        p3 = self.smooth_conv_p3(p3)

        out = [p3, p4, p5]  # self.concat((p3, p4, p5))

        return out

    def Xavier_conv(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='same',
                    has_bias=False):
        init_value = XavierUniform()
        return nn.Conv2d(in_channels, out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         pad_mode=pad_mode, weight_init=init_value, has_bias=has_bias)


class PSEFPN(nn.Cell):
    """
    PSE Feature Pyramid Network (FPN) module for text detection.

    This module takes multiple input feature maps and performs feature fusion
    and upsampling to generate a single output feature map.

    Args:
        in_channels (List[int]): The input channel dimensions for each feature map
                                 in the following order: [c2, c3, c4, c5].
        out_channels (int): The output channel size.

    Returns:
        Tensor: The output feature map of shape [batch_size, out_channels * 4, H, W].

    """

    def __init__(self, in_channels: List[int], out_channels):
        super().__init__()
        super(PSEFPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # reduce layers
        self.reduce_conv_c2 = _conv(in_channels[0], out_channels, kernel_size=1, has_bias=True)
        self.reduce_bn_c2 = _bn(out_channels)
        self.reduce_relu_c2 = nn.ReLU()

        self.reduce_conv_c3 = _conv(in_channels[1], out_channels, kernel_size=1, has_bias=True)
        self.reduce_bn_c3 = _bn(out_channels)
        self.reduce_relu_c3 = nn.ReLU()

        self.reduce_conv_c4 = _conv(in_channels[2], out_channels, kernel_size=1, has_bias=True)
        self.reduce_bn_c4 = _bn(out_channels)
        self.reduce_relu_c4 = nn.ReLU()

        self.reduce_conv_c5 = _conv(in_channels[3], out_channels, kernel_size=1, has_bias=True)
        self.reduce_bn_c5 = _bn(out_channels)
        self.reduce_relu_c5 = nn.ReLU()

        # smooth layers
        self.smooth_conv_p4 = _conv(out_channels, out_channels, kernel_size=3, has_bias=True)
        self.smooth_bn_p4 = _bn(out_channels)
        self.smooth_relu_p4 = nn.ReLU()

        self.smooth_conv_p3 = _conv(out_channels, out_channels, kernel_size=3, has_bias=True)
        self.smooth_bn_p3 = _bn(out_channels)
        self.smooth_relu_p3 = nn.ReLU()

        self.smooth_conv_p2 = _conv(out_channels, out_channels, kernel_size=3, has_bias=True)
        self.smooth_bn_p2 = _bn(out_channels)
        self.smooth_relu_p2 = nn.ReLU()

        self._resize_bilinear = nn.ResizeBilinear()

        self.concat = ops.Concat(axis=1)

    def construct(self, features):
        assert len(features) == 4, f"PSENet receives 4 levels features instead of {len(features)} levels of features."
        c2, c3, c4, c5 = features
        p5 = self.reduce_conv_c5(c5)
        p5 = self.reduce_relu_c5(self.reduce_bn_c5(p5))

        c4 = self.reduce_conv_c4(c4)
        c4 = self.reduce_relu_c4(self.reduce_bn_c4(c4))
        p4 = self._resize_bilinear(p5, scale_factor=2) + c4
        p4 = self.smooth_conv_p4(p4)
        p4 = self.smooth_relu_p4(self.smooth_bn_p4(p4))

        c3 = self.reduce_conv_c3(c3)
        c3 = self.reduce_relu_c3(self.reduce_bn_c3(c3))
        p3 = self._resize_bilinear(p4, scale_factor=2) + c3
        p3 = self.smooth_conv_p3(p3)
        p3 = self.smooth_relu_p3(self.smooth_bn_p3(p3))

        c2 = self.reduce_conv_c2(c2)
        c2 = self.reduce_relu_c2(self.reduce_bn_c2(c2))
        p2 = self._resize_bilinear(p3, scale_factor=2) + c2
        p2 = self.smooth_conv_p2(p2)
        p2 = self.smooth_relu_p2(self.smooth_bn_p2(p2))

        p3 = self._resize_bilinear(p3, scale_factor=2)
        p4 = self._resize_bilinear(p4, scale_factor=4)
        p5 = self._resize_bilinear(p5, scale_factor=8)

        out = self.concat((p2, p3, p4, p5))

        return out


class EASTFPN(nn.Cell):
    def __init__(self, in_channels, out_channels=128):
        super(EASTFPN, self).__init__()
        self.in_channels = in_channels[::-1]  # self.in_channels: [2048, 1024, 512, 256]
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(self.in_channels[0] + self.in_channels[1], self.in_channels[0] // 4, 1, has_bias=True)
        self.bn1 = nn.BatchNorm2d(self.in_channels[0] // 4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            self.in_channels[0] // 4,
            self.in_channels[0] // 4,
            3,
            padding=1,
            pad_mode='pad',
            has_bias=True)
        self.bn2 = nn.BatchNorm2d(self.in_channels[0] // 4)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            self.in_channels[0] // 4 + self.in_channels[2], self.in_channels[1] // 4, 1, has_bias=True)
        self.bn3 = nn.BatchNorm2d(self.in_channels[1] // 4)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(
            self.in_channels[1] // 4,
            self.in_channels[1] // 4,
            3,
            padding=1,
            pad_mode='pad',
            has_bias=True)
        self.bn4 = nn.BatchNorm2d(self.in_channels[1] // 4)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(self.in_channels[1] // 4 + self.in_channels[3], self.in_channels[2] // 4, 1)
        self.bn5 = nn.BatchNorm2d(self.in_channels[2] // 4)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(
            self.in_channels[2] // 4,
            self.in_channels[2] // 4,
            3,
            padding=1,
            pad_mode='pad',
            has_bias=True)
        self.bn6 = nn.BatchNorm2d(self.in_channels[2] // 4)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(
            self.in_channels[2] // 4,
            self.out_channels,
            3,
            padding=1,
            pad_mode='pad',
            has_bias=True)
        self.bn7 = nn.BatchNorm2d(self.out_channels)
        self.relu7 = nn.ReLU()
        self.concat = ops.Concat(axis=1)

    def construct(self, features):
        f1, f2, f3, f4 = features

        out = ops.ResizeBilinearV2(True)(f4, f3.shape[2:])
        out = self.concat((out, f3))
        out = self.relu1(self.bn1(self.conv1(out)))
        out = self.relu2(self.bn2(self.conv2(out)))

        out = ops.ResizeBilinearV2(True)(out, f2.shape[2:])
        out = self.concat((out, f2))
        out = self.relu3(self.bn3(self.conv3(out)))
        out = self.relu4(self.bn4(self.conv4(out)))

        out = ops.ResizeBilinearV2(True)(out, f1.shape[2:])
        out = self.concat((out, f1))
        out = self.relu5(self.bn5(self.conv5(out)))
        out = self.relu6(self.bn6(self.conv6(out)))

        out = self.relu7(self.bn7(self.conv7(out)))
        return out
