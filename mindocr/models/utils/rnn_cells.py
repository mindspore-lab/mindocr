"""RNN Cells that supports FP16 inputs
"""
import mindspore.ops as P
from mindspore.ops.primitive import constexpr
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

@constexpr
def _check_batch_size_equal(batch_size_x, batch_size_hx, cls_name):
    if batch_size_x != batch_size_hx:
        raise ValueError(f"For '{cls_name}' batch size of x and hx must be equal, but got {batch_size_x} of x "
                         f"and {batch_size_hx} of hx.")


class GRUCell(RNNCellBase):
    r"""
    A GRU(Gated Recurrent Unit) cell.

    .. math::

        \begin{array}{ll}
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \end{array}

    Here :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product. :math:`W, b`
    are learnable weights between the output and the input in the formula. For instance,
    :math:`W_{ir}, b_{ir}` are the weight and bias used to transform from input :math:`x` to :math:`r`.
    Details can be found in paper
    `Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
    <https://aclanthology.org/D14-1179.pdf>`_.

    The LSTMCell can be simplified in NN layer, the following formula:

    .. math::
        h^{'},c^{'} = LSTMCell(x, (h_0, c_0))

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        has_bias (bool): Whether the cell has bias `b_in` and `b_hn`. Default: True.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch_size, `input_size`).
        - **hx** (Tensor) - Tensor of data type mindspore.float32 and shape (batch_size, `hidden_size`).
          Data type of `hx` must be the same as `x`.

    Outputs:
        - **hx'** (Tensor) - Tensor of shape (batch_size, `hidden_size`).

    Raises:
        TypeError: If `input_size`, `hidden_size` is not an int.
        TypeError: If `has_bias` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.GRUCell(10, 16)
        >>> x = Tensor(np.ones([5, 3, 10]).astype(np.float32))
        >>> hx = Tensor(np.ones([3, 16]).astype(np.float32))
        >>> output = []
        >>> for i in range(5):
        ...     hx = net(x[i], hx)
        ...     output.append(hx)
        >>> print(output[0].shape)
        (3, 16)
    """
    def __init__(self, input_size: int, hidden_size: int, has_bias: bool = True):
        super().__init__(input_size, hidden_size, has_bias, num_chunks=3)

    def construct(self, x, hx):
        _check_batch_size_equal(x.shape[0], hx.shape[0], self.cls_name)

        # FIX: make sure the weight and bias dtype is same as the data type from x
        # prevent the input type inconsistent error from P.MatMul operator
        weight_ih = P.cast(self.weight_ih, x.dtype)
        weight_hh = P.cast(self.weight_hh, x.dtype)
        bias_ih = P.cast(self.bias_ih, x.dtype)
        bias_hh = P.cast(self.bias_hh, x.dtype)

        return _gru_cell(x, hx, weight_ih, weight_hh, bias_ih, bias_hh)
