from mindspore import nn

from mindformers.models import SAMImageEncoder


class SAMEncoder(SAMImageEncoder):
    """SAM encoder for Vary system"""
    def __init__(self, config) -> None:
        super().__init__(config)
        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, pad_mode="pad", padding=1, has_bias=False)
        self.net_3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, pad_mode="pad", padding=1, has_bias=False)

    def construct(self, x):
        x = super().construct(x)
        x = self.net_2(x)
        x = self.net_3(x)
        x = x.flatten(start_dim=2).permute(0, 2, 1)
        return x
