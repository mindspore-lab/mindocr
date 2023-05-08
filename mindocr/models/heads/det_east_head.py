from mindspore import nn, ops


class EASTHead(nn.Cell):
    def __init__(self, in_channels, scope=512):
        super(EASTHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(in_channels, 4, 1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv3 = nn.Conv2d(in_channels, 1, 1)
        self.sigmoid3 = nn.Sigmoid()
        self.scope = scope
        self.concat = ops.Concat(axis=1)
        self.PI = 3.1415926535898

    def construct(self, x):
        score = self.sigmoid1(self.conv1(x))
        pred = {'score': score}
        loc = self.sigmoid2(self.conv2(x)) * self.scope
        angle = (self.sigmoid3(self.conv3(x)) - 0.5) * self.PI
        geo = self.concat((loc, angle))
        pred['geo'] = geo
        return pred
