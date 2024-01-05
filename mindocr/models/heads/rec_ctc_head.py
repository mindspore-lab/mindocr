import math
from typing import Optional

import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import Uniform

__all__ = ['CTCHead']


def crnn_head_initialization(k):
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = Uniform(stdv)
    return initializer


class CTCHead(nn.Cell):
    """An MLP head for CTC Loss.
    The MLP encodes and classifies the features, then return a logit tensor in shape [W, BS, num_classes]

    Args:
        in_channels: Number of the input channels.
        out_channels: Number of the output channels.
        mid_channels: Apply a dense layer of the given number of the channels before output. If it is none,
            then it is not applied. Default: None.
        weight_init: Weight initialzation method. Can be "normal" or "crnn_customised". Default: normal.
        bias_init: Bias initialization methdo. Can be "zeros" or "crnn_customised". Default: zeros.
        dropout: Dropout value in the head. Default: 0.

    Inputs:
        x (Tensor): feature in shape [W, BS, C]

    Outputs:
        h (Tensor): if training, h is logits in shape [W, BS, num_classes], where W - sequence len is fixed.
            if not training, h is class probabilites in shape [BS, W, num_classes].
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels: Optional[int] = None,
                 weight_init: str = 'normal',
                 bias_init: str = 'zeros',
                 dropout: float = 0.) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.mid_channels = mid_channels

        if weight_init == "crnn_customised":
            weight_init = crnn_head_initialization(in_channels)

        if bias_init == "crnn_customised":
            bias_init = crnn_head_initialization(in_channels)

        if mid_channels is None:
            self.dense1 = nn.Dense(in_channels, out_channels, weight_init=weight_init, bias_init=bias_init)
        else:
            self.dense1 = nn.Dense(in_channels, mid_channels, weight_init=weight_init, bias_init=bias_init)
            self.dropout = nn.Dropout(p=dropout)
            self.dense2 = nn.Dense(mid_channels, out_channels, weight_init=weight_init, bias_init=bias_init)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        h = self.dense1(x)
        if self.mid_channels is not None:
            h = self.dropout(h)
            h = self.dense2(h)

        if not self.training:
            h = ops.Softmax(axis=2)(h)
            h = h.transpose((1, 0, 2))

        return h
