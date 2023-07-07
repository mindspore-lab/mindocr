from typing import List, Optional

import numpy as np

from mindspore import Tensor, nn, ops

__all__ = ['RNNEncoder']


class RNNEncoder(nn.Cell):
    """Sequence encoder which contains reshape and bidirectional LSTM layers.
    Receive visual features [N, C, 1, W] and reshape it to shape [W, N, C].
    Use Bi-LSTM to encode into new features in shape [W, N, C]. where W - seq len,
        N - batch size, C - feature len

    Args:
        input_channels (int):  C, number of input channels, corresponding to feature length
        hidden_size(int): the hidden size in LSTM layers, Default: 512
        batch_size: Batch size. Default: None

    Inputs:
        x (Tensor): feature, a Tensor of shape `(N, C, 1, W)`.
            Note that H must be 1. C - input channels can be viewed as feature length for each time step.
            N is batch size.

    Outputs:
        Tensor: Encoded features, in shape with (W, N, 2 * C)
    """

    def __init__(self, in_channels: int, hidden_size: int = 512,  batch_size: Optional[int] = None) -> None:
        super().__init__()
        self.out_channels = 2 * hidden_size

        self.seq_encoder = nn.LSTM(input_size=in_channels,
                                   hidden_size=hidden_size,
                                   num_layers=2,
                                   has_bias=True,
                                   dropout=0.,
                                   bidirectional=True)

        self.hx = None
        if batch_size is not None:
            h0 = Tensor(np.zeros([2 * 2, batch_size, hidden_size]).astype(np.float32))
            c0 = Tensor(np.zeros([2 * 2, batch_size, hidden_size]).astype(np.float32))
            self.hx = (h0, c0)

    def construct(self, features: List[Tensor]) -> Tensor:
        x = features[0]
        x = ops.squeeze(x, axis=2)  # [N, C, W]
        x = ops.transpose(x, (2, 0, 1))  # [W, N, C]

        if self.hx is None:
            x, _ = self.seq_encoder(x)
        else:
            x, _ = self.seq_encoder(x, self.hx)

        return x
