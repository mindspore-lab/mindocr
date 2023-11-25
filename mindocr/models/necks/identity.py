from mindspore import nn

__all__ = ["Identity"]


class Identity(nn.Cell):
    """
    select feature from the backbone output.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.out_channels = in_channels

    def construct(self, x):
        return x
