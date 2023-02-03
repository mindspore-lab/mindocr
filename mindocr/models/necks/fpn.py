import mindspore as ms
from mindspore import nn

class FPN(nn.Cell):
    def __init__(self, in_channels, out_channels=256, **kwargs):
        super().__init__()
        self.out_channels = out_channels

    def construct(self, x):
        x1, x2, x3, x4 = x
        return x1
