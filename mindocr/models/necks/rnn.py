import numpy as np

import mindspore as ms
from mindspore import Tensor, nn, ops

__all__ = ['RNNEncoder']


# TODO: check mindspore nn LSTM diff in performance and precision from paddle/pytorch
# TODO: what is the initialization method by default?
class RNNEncoder(nn.Cell):
    """
     CRNN sequence encoder which contains reshape and bidirectional LSTM layers.
     Receive visual features [N, C, 1, W]
     Reshape features to shape [W, N, C]
     Use Bi-LSTM to encode into new features in shape [W, N, 2*C].
     where W - seq len, N - batch size, C - feature len

     Args:
        input_channels (int):  C, number of input channels, corresponding to feature length
        hidden_size(int): the hidden size in LSTM layers, default is 512
     """

    def __init__(self, in_channels, hidden_size=512, batch_size=None):
        super().__init__()
        self.out_channels = 2 * hidden_size

        self.seq_encoder = nn.LSTM(input_size=in_channels,
                                   hidden_size=hidden_size,
                                   num_layers=2,
                                   has_bias=True,
                                   dropout=0.,
                                   bidirectional=True)

        # TODO: do we need to add batch size to compute hx menioned in MindSpore LSTM doc
        self.hx = None
        if batch_size is not None:
            h0 = Tensor(np.zeros([2 * 2, batch_size, hidden_size]).astype(np.float32))
            c0 = Tensor(np.zeros([2 * 2, batch_size, hidden_size]).astype(np.float32))
            self.hx = (h0, c0)

    def construct(self, features):
        """
        Args:
            x (Tensor): feature, a Tensor of shape :math:`(N, C, 1, W)`.
                Note that H must be 1. Width W can be viewed as time length in CRNN algorithm.
                C - input channels can be viewed as feature length for each time step.  N is batch size.

        Returns:
            Tensor: Encoded features . Shape :math:`(W, N, 2*C)` where
        """
        x = features[0]
        assert x.shape[2] == 1, f'Feature height must be 1, but got {x.shape[2]} from x.shape {x.shape}'
        x = ops.squeeze(x, axis=2)  # [N, C, W]
        x = ops.transpose(x, (2, 0, 1))  # [W, N, C]

        if self.hx is None:
            x, hx_n = self.seq_encoder(x)
        else:
            print('using self.hx')
            x, hx_n = self.seq_encoder(x, self.hx)  # the results are the same

        return x


# TODO: check correctness, this structure is different from paddle
'''
class BidirectionalLSTM(nn.Cell):

    def __init__(self, nIn, nHidden, nOut, batch_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Dense(in_channels=nHidden * 2, out_channels=nOut)
        self.h0 = Tensor(np.zeros([1 * 2, batch_size, nHidden]).astype(np.float32))
        self.c0 = Tensor(np.zeros([1 * 2, batch_size, nHidden]).astype(np.float32))

    def construct(self, x):
        recurrent, _ = self.rnn(x, (self.h0, self.c0))
        T, b, h = P.Shape()(recurrent)
        t_rec = P.Reshape()(recurrent, (T * b, h,))

        out = self.embedding(t_rec)  # [T * b, nOut]
        out = P.Reshape()(out, (T, b, -1,))

        return out
'''

if __name__ == '__main__':
    from mindocr.utils.debug import initialize_network_with_constant

    c, h, w = 128, 1, 16
    bs = 8
    x = ms.Tensor(np.random.rand(bs, c, h, w), dtype=ms.float32)

    model1 = RNNEncoder(in_channels=c, hidden_size=256, batch_size=8)
    initialize_network_with_constant(model1, c_weight=1.0)
    model2 = RNNEncoder(in_channels=c, hidden_size=256, batch_size=None)
    initialize_network_with_constant(model2, c_weight=1.0)

    h1 = model1.construct(x)
    h2 = model2.construct(x)

    print('w/ hx', h1.shape, h1.sum())
    print('w/o hx', h2.shape, h2.sum())
