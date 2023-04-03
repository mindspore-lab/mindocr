"""Identity Module"""
from mindspore import nn


class Identity(nn.Cell):
    """Identity"""

    def construct(self, x):
        return x
