import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as ops


class BiLSTM(nn.Cell):

    def __init__(self, in_channels,
                 hidden_size,
                 batch_size,
                 out_channels=None,
                 num_layers=1,
                 batch_first=True,
                 dropout=0.0,
                 bidirectional=True,
                 ):
        super(BiLSTM, self).__init__()
        self.out_channels = out_channels
        self.rnn = nn.LSTM(in_channels, hidden_size, num_layers=num_layers,
                           batch_first=batch_first, dropout=dropout,
                           bidirectional=bidirectional)


        num_directions = 2 if bidirectional else 1
        self.h0 = Tensor(np.zeros(
            [num_directions * num_layers, batch_size, hidden_size]).astype(np.float32))
        self.c0 = Tensor(np.zeros(
            [num_directions * num_layers, batch_size, hidden_size]).astype(np.float32))
        
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def construct(self, x):
        # x: (batch_size, seq_len, input_size)
        x, _ = self.rnn(x, (self.h0, self.c0))
        return x
