"""
Custom operators.
"""

from mindspore import nn

__all__ = ["Swish"]


class Swish(nn.Cell):
    """
    Swish activation function: x * sigmoid(x).

    Args:
        None

    Return:
        Tensor

    Example:
        >>> x = Tensor(((20, 16), (50, 50)), mindspore.float32)
        >>> Swish()(x)
    """

    def __init__(self):
        super().__init__()
        self.result = None
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        result = x * self.sigmoid(x)
        return result
