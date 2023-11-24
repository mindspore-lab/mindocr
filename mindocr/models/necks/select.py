from mindspore import nn

__all__ = ['Select']


class Select(nn.Cell):
    """
    select feature from the backbone output.
    """
    def __init__(self, in_channels, index=-1):
        super().__init__()
        self.index = index
        self.out_channels = in_channels[index]

    def construct(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            return x[self.index]
        else:
            return x
