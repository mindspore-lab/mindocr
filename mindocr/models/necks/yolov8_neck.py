from mindspore import nn

__all__ = ['YOLOv8Neck']


class YOLOv8Neck(nn.Cell):
    """
    select feature from the backbone output.
    """
    def __init__(self, in_channels, index=None):
        super().__init__()
        self.index = index
        self.out_channels = in_channels

    def construct(self, x):
        if isinstance(x, (list, tuple)) and self.index is not None:
            x = [x[_] for _ in self.index]
        return x
