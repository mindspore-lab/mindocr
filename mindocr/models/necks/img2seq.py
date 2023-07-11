from typing import List

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

__all__ = ["Img2Seq"]


class Img2Seq(nn.Cell):
    """Img2Seq Neck, for converting the input from [N, C, 1, W] to [W, N, C] for
    further processing.

    Args:
        in_channels: Number of the input channels.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.out_channels = in_channels

    def construct(self, features: List[Tensor]) -> Tensor:
        x = features[0]
        x = ops.squeeze(x, axis=2)
        x = ops.transpose(x, (2, 0, 1))
        return x
