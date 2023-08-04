from typing import List, Union

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

__all__ = ["Img2Seq"]


class Img2Seq(nn.Cell):
    """Img2Seq Neck, for converting the list of input from [N, W, C] to [W, N, C] for
    further processing.

    Args:
        in_channels: Number of the input channels.
        select_inds: The index list of the input that need to be transformed. Default: [-1]
    """

    def __init__(self, in_channels: List[int], select_inds: List[int] = [-1]) -> None:
        super().__init__()
        self.out_channels = [in_channels[x] for x in select_inds]
        if len(self.out_channels) == 1:
            self.out_channels = self.out_channels[0]
        self.select_inds = select_inds

    def construct(self, features: List[Tensor]) -> Union[Tensor, List[Tensor]]:
        output = list()
        for ind in self.select_inds:
            x = features[ind]
            x = ops.transpose(x, (1, 0, 2))
            output.append(x)

        if len(self.select_inds) == 1:
            output = output[0]
        return output
