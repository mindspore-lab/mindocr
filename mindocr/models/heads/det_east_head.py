from mindspore import nn, ops
from math import pi


class EASTHead(nn.Cell):
    def __init__(self, in_channels, scope=512):
        super(EASTHead, self).__init__()
        self.branch1_det_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1,
                                           padding=1, pad_mode='pad',
                                           has_bias=True)
        self.branch1_det_bn1 = nn.BatchNorm2d(128)
        self.branch1_det_relu1 = nn.ReLU()

        self.branch1_det_conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1,
                                           pad_mode='pad',
                                           has_bias=True)
        self.branch1_det_bn2 = nn.BatchNorm2d(64)
        self.branch1_det_relu2 = nn.ReLU()

        self.branch2_det_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1,
                                           padding=1,
                                           pad_mode='pad',
                                           has_bias=True)
        self.branch2_det_bn1 = nn.BatchNorm2d(128)
        self.branch2_det_relu1 = nn.ReLU()

        self.branch2_det_conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1,
                                           pad_mode='pad',
                                           has_bias=True)
        self.branch2_det_bn2 = nn.BatchNorm2d(64)
        self.branch2_det_relu2 = nn.ReLU()

        self.branch3_det_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1,
                                           padding=1,
                                           pad_mode='pad',
                                           has_bias=True)
        self.branch3_det_bn1 = nn.BatchNorm2d(128)
        self.branch3_det_relu1 = nn.ReLU()

        self.branch3_det_conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1,
                                           pad_mode='pad',
                                           has_bias=True)
        self.branch3_det_bn2 = nn.BatchNorm2d(64)
        self.branch3_det_relu2 = nn.ReLU()

        self.conv1 = nn.Conv2d(64, 1, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(64, 4, 1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv3 = nn.Conv2d(64, 1, 1)
        self.sigmoid3 = nn.Sigmoid()
        self.scope = scope
        self.concat = ops.Concat(axis=1)
        self.PI = pi

    def construct(self, x):
        x1 = self.branch1_det_relu1(self.branch1_det_bn1(self.branch1_det_conv1(x)))
        x1 = self.branch1_det_relu2(self.branch1_det_bn2(self.branch1_det_conv2(x1)))

        x2 = self.branch2_det_relu1(self.branch2_det_bn1(self.branch2_det_conv1(x)))
        x2 = self.branch2_det_relu2(self.branch2_det_bn2(self.branch2_det_conv2(x2)))

        x3 = self.branch3_det_relu1(self.branch3_det_bn1(self.branch3_det_conv1(x)))
        x3 = self.branch3_det_relu2(self.branch3_det_bn2(self.branch3_det_conv2(x3)))

        score = self.sigmoid1(self.conv1(x1))
        pred = {'score': score}
        loc = self.sigmoid2(self.conv2(x2)) * self.scope
        angle = (self.sigmoid3(self.conv3(x3)) - 0.5) * self.PI
        geo = self.concat((loc, angle))
        pred['geo'] = geo
        return pred
