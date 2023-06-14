from typing import List

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

__all__ = ["Img2Seq"]


class Img2Seq(nn.Cell):
    """
    Inputs: feature list with shape [N, C, 1, W]
    Outputs: first feature with shape [W, N, C]
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.out_channels = in_channels

    def construct(self, features: List[Tensor]) -> Tensor:
        x = features[0]
        x = ops.squeeze(x, axis=2)
        x = ops.transpose(x, (2, 0, 1))
        return x
