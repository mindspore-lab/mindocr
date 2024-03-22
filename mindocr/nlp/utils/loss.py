from mindspore import Tensor, nn, ops
from mindspore.common import dtype as mstype

__all__ = ["CrossEntropyLoss"]


class _Softmax(nn.Cell):
    """
    Calculate the softmax results with given logits. The bprop of the cell is rewritten,
    just returns the accepted dout as returns. This cell should be used together with _NLLoss,
    to optimize the bprop of the cross entroy loss.

    Inputs:
        - **logits** (Tensor) - Tensor of shape (N, C). Data type must be float16 or float32. The output logits of
          the backbone.

        - **label** (Tensor) - Tensor of shape (N, 1). The ground truth label of the sample.

    Returns:
        The corresponding softmax results.
    """

    def __init__(self):
        super(_Softmax, self).__init__()
        # on/off value for onehot, for smooth labeling, modify the off_value
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)

        self.sum = ops.ReduceSum()
        self.max = ops.ArgMaxWithValue(axis=-1, keep_dims=True)
        self.sub = ops.Sub()
        self.exp = ops.Exp()
        self.div = ops.RealDiv()
        self.onehot = ops.OneHot()

    def construct(self, logits, label):
        """Forward process"""
        # LogSoftmax for logits over last dimension
        logits = ops.cast(logits, mstype.float32)
        _, logit_max = self.max(logits)
        logit_sub = self.sub(logits, logit_max)
        logit_exp = self.exp(logit_sub)
        exp_sum = self.sum(logit_exp, -1)
        exp_sum = ops.Reshape()(exp_sum, (ops.shape(exp_sum)[0], 1))
        softmax_result = self.div(logit_exp, exp_sum)

        one_hot_label = self.onehot(label, ops.shape(logits)[-1], self.on_value, self.off_value)
        return softmax_result, one_hot_label

    def bprop(self, logits, label, _, dout):
        """just return the loss of the dout. Note this should be used together with _NLLLoss"""
        d_logits = ops.cast(dout[0], ops.dtype(logits))
        return d_logits, ops.zeros_like(label)


class _NLLLoss(nn.Cell):
    """
    Calculate the NLLLoss results with given softmax results and the label. The bprop of the cell is rewritten.
    This cell should be used together with _Softmax, to optimize the bprop of the cross entroy loss.

    Inputs:
        - **softmax_result** (Tensor) - Tensor of shape (N, C). Data type is float32.
        - **one_hot_label** (Tensor) - Tensor of shape (N, C). The ground truth label in one-hot format of the sample.

    Returns:
        The corresponding loss results.
    """

    def __init__(self, eps_const=1e-24):
        super(_NLLLoss, self).__init__()
        self.repeat_loss = 1
        self.eps_const = Tensor(eps_const, mstype.float32)
        self.sum = ops.ReduceSum()
        self.mul = ops.Mul()
        self.neg = ops.Neg()
        self.log = ops.Log()
        self.add = ops.Add()

    def construct(self, softmax_result, one_hot_label):
        log_softmax_result = self.log(self.add(softmax_result, self.eps_const))
        loss = self.mul(log_softmax_result, one_hot_label)
        loss_unsum = self.neg(loss)
        loss_reduce = self.sum(loss_unsum, -1)
        return loss_reduce

    def bprop(self, softmax_result, one_hot_label, _, dout):
        """A simplified function. Note this should be used together with _Softmax"""
        logits = softmax_result - one_hot_label
        logits = logits * ops.ExpandDims()(dout, -1) * self.repeat_loss

        return logits, ops.zeros_like(one_hot_label)


class CrossEntropyLoss(nn.Cell):
    """
    Calculate the cross entropy loss.

    Inputs:
        - **logits** (Tensor) - Tensor of shape (N, C). Data type must be float16 or float32. The output logits of
          the backbone.

        - **labels** (Tensor) - Tensor of shape (N, ). The ground truth label of the sample.

        - **input_mask** (Tensor) - Tensor of shape (N, ). input_mask indicates whether there are padded inputs and for
          padded inputs it will not be counted into loss.

    Returns:
        The corresponding cross entropy loss.
    """

    def __init__(self, eps_const=1e-24):
        super(CrossEntropyLoss, self).__init__()
        self.enable_force_redistribute = False
        self.sum2 = ops.ReduceSum()
        self.mul2 = ops.Mul()
        self.add2 = ops.Add()
        self.div2 = ops.RealDiv()
        self.relu = ops.ReLU()

        self._softmax = _Softmax()
        self._nllloss = _NLLLoss(eps_const)

    def construct(self, logits, label, input_mask):
        """Forward process"""
        # The add is used for forcing the redistribution before stepping in sub graphs, when semi/auto parallel enabled.
        if self.enable_force_redistribute:
            logits = self.add(logits, 0)
            label = self.add_label(label, 0)
        softmax, one_hot_label = self._softmax(logits, label)
        loss_reduce = self._nllloss(softmax, one_hot_label)

        # Using input_mask to mask the loss
        input_mask = ops.Reshape()(input_mask, (-1,))
        numerator = self.sum2(self.mul2(loss_reduce, input_mask))

        denominator = self.add2(self.sum2(input_mask), ops.Cast()(ops.tuple_to_array((1e-5,)), mstype.float32))
        loss = self.div2(numerator, denominator)

        return loss
