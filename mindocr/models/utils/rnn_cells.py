"""RNN Cells that supports FP16 inputs
"""
import mindspore.ops as P
from mindspore.nn.layer.rnn_cells import RNNCellBase

__all__ = ["GRUCell"]


def _gru_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh):
    '''GRU cell function'''
    if b_ih is None:
        gi = P.MatMul(False, True)(inputs, w_ih)
        gh = P.MatMul(False, True)(hidden, w_hh)
    else:
        gi = P.MatMul(False, True)(inputs, w_ih) + b_ih
        gh = P.MatMul(False, True)(hidden, w_hh) + b_hh
    i_r, i_i, i_n = P.Split(1, 3)(gi)
    h_r, h_i, h_n = P.Split(1, 3)(gh)

    resetgate = P.Sigmoid()(i_r + h_r)
    inputgate = P.Sigmoid()(i_i + h_i)
    newgate = P.Tanh()(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)

    return hy


class GRUCell(RNNCellBase):
    """A Modified version of RU(Gated Recurrent Unit) cell, based on mindspore.nn.layer.rnn_cells.GRUCell
    It adds a type cast protection of all initial variable
    """
    def __init__(self, input_size: int, hidden_size: int, has_bias: bool = True):
        super().__init__(input_size, hidden_size, has_bias, num_chunks=3)

    def construct(self, x, hx):
        # FIX: make sure the weight and bias dtype is same as the data type from x
        # prevent the input type inconsistent error from P.MatMul operator
        weight_ih = P.cast(self.weight_ih, x.dtype)
        weight_hh = P.cast(self.weight_hh, x.dtype)
        bias_ih = P.cast(self.bias_ih, x.dtype)
        bias_hh = P.cast(self.bias_hh, x.dtype)

        return _gru_cell(x, hx, weight_ih, weight_hh, bias_ih, bias_hh)
