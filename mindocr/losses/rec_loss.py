import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore import Tensor
from mindspore.nn.loss.loss import LossBase
from mindspore.ops import operations as P


__all__ = ['CTCLoss', 'AttentionLoss']


_neg = P.Neg()
_gather_d = P.GatherD()
_gather = P.Gather()
_ones_like = P.OnesLike()
_equal = P.Equal()
_exp = P.Exp()
_reduce_sum = P.ReduceSum(True)
_log = P.Log()


class CTCLoss(LossBase):
    """
    CTCLoss definition

    Args:
        pred_seq_len(int): the length of the predicted character sequence. For text images, this value equals to W - the width of feature map encoded by the visual bacbkone. This can be obtained by probing the output shape in the network.
            E.g., for a training image in shape (3, 32, 100), the feature map encoded by resnet34 bacbkone is in shape (512, 1, 4), W = 4, sequence len is 4.
        max_label_len(int): the maximum number of characters in a text label, i.e. max_text_len in yaml.
        batch_size(int): batch size of input logits. bs
    """

    def __init__(self, pred_seq_len: int = 26, max_label_len: int = 25, batch_size: int = 32, reduction: str = 'mean') -> None:
        super(CTCLoss, self).__init__(reduction=reduction)
        assert pred_seq_len > max_label_len, 'pred_seq_len is required to be larger than max_label_len for CTCLoss. Please adjust the strides in the backbone, or reduce max_text_length in yaml'
        self.sequence_length = Tensor(np.array([pred_seq_len] * batch_size), ms.int32)

        label_indices = []
        for i in range(batch_size):
            for j in range(max_label_len):
                label_indices.append([i, j])
        self.label_indices = Tensor(np.array(label_indices), ms.int64)
        self.ctc_loss = ops.CTCLoss(ctc_merge_repeated=True)

    def construct(self, pred: Tensor, label: Tensor) -> Tensor:
        '''
        Args:
            pred (Tensor): network prediction which is a
                logit Tensor in shape (W, BS, NC), where W - seq len, BS - batch size. NC - num of classes (types of character + blank + 1)
            label (Tensor): GT sequence of character indices in shape (BS, SL), SL - sequence length, which is padded to max_text_length
        Returns:
            loss value (Tensor)
        '''
        logit = pred
        label_values = ops.reshape(label, (-1,))

        loss, _ = self.ctc_loss(logit, self.label_indices, label_values, self.sequence_length)
        loss = self.get_loss(loss)
        return loss


class AttentionLoss(LossBase):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        # ignore <GO> symbol, assume it is placed at 0th index
        self.criterion = CrossEntropyLoss(reduction=reduction, ignore_index=0)

    def construct(self, logits: Tensor, labels: Tensor) -> Tensor:
        labels = labels[:, 1:]  # wihout <GO> symbol
        num_classes = logits.shape[-1]
        logits = ops.reshape(logits, (-1, num_classes))
        labels = ops.reshape(labels, (-1,))
        return self.criterion(logits, labels)


class CrossEntropyLoss(LossBase):
    def __init__(self, weight=None, ignore_index=-100, reduction='mean',
                 label_smoothing=0.0):
        super().__init__(reduction)
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def construct(self, logits, labels):
        return cross_entropy(logits, labels, self.weight, self.ignore_index, self.reduction, self.label_smoothing)


def cross_entropy(inputs, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    class_dim = 0 if inputs.ndim == 1 else 1
    return nll_loss(_innner_log_softmax(inputs, class_dim), target, weight, ignore_index, reduction, label_smoothing)


def nll_loss(inputs, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    ndim = inputs.ndim
    if ndim == 2:
        ret = _nll_loss(inputs, target, -1, weight, ignore_index, reduction, label_smoothing)
    elif ndim == 4:
        ret = _nll_loss(inputs, target, 1, weight, ignore_index, reduction, label_smoothing)
    elif ndim == 1:
        ret = _nll_loss(inputs, target, 0, weight, ignore_index, reduction, label_smoothing)
    else:
        n = inputs.shape[0]
        c = inputs.shape[1]
        out_size = (n,) + inputs.shape[2:]
        inputs = inputs.view(n, c, 1, -1)
        target = target.view(n, 1, -1)
        if reduction != 'none':
            ret = _nll_loss(inputs, target, 1, weight, ignore_index, reduction, label_smoothing)
        else:
            ret = _nll_loss(inputs, target, 1, weight, ignore_index, label_smoothing=label_smoothing)
            ret = ret.view(out_size)
    return ret


def _nll_loss(inputs, target, target_dim=-1, weight=None, ignore_index=None, reduction='none', label_smoothing=0.0):
    """nll loss inner function"""
    if target.ndim == inputs.ndim - 1:
        target = target.expand_dims(target_dim)
    if ignore_index is not None:
        non_pad_mask = _equal(target, ignore_index)
        target = target.masked_fill(non_pad_mask, 0)
    else:
        non_pad_mask = target
    loss = _neg(_gather_d(inputs, target_dim, target))
    smooth_loss = _neg(inputs.sum(axis=target_dim, keepdims=True))
    if weight is not None:
        loss_weights = _gather(weight, target, 0)
        loss = loss * loss_weights
    else:
        loss_weights = _ones_like(loss)
    if ignore_index is not None:
        loss = loss.masked_fill(non_pad_mask, 0.)
        loss_weights = loss_weights.masked_fill(non_pad_mask, 0.)
        smooth_loss = smooth_loss.masked_fill(non_pad_mask, 0.)

    loss = loss.squeeze(target_dim)
    smooth_loss = smooth_loss.squeeze(target_dim)

    if reduction == 'sum':
        loss = loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == 'mean':
        loss = loss.sum() / loss_weights.sum()
        smooth_loss = smooth_loss.mean()

    eps_i = label_smoothing / inputs.shape[target_dim]
    loss = (1. - label_smoothing) * loss + eps_i * smooth_loss

    return loss


def _innner_log_softmax(inputs, axis):
    """inner implementation of log_softmax, since the LogSoftmaxGrad op do not support inputs > 2d"""
    return inputs - logsumexp(inputs, axis)


def logsumexp(x, axis):
    x_max = x.max(axis=axis, keepdims=True)
    x_exp = _exp(x - x_max)
    x_sumexp = _reduce_sum(x_exp, axis)
    x_logsumexp = _log(x_sumexp)
    return x_logsumexp + x_max
