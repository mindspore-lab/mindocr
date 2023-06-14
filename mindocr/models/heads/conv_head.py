from mindspore import nn


class ConvHead(nn.Cell):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.SequentialCell(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def construct(self, x):
        return self.conv(x)
