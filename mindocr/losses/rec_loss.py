import numpy as np

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.nn.loss.loss import LossBase

__all__ = ["CTCLoss", "AttentionLoss", "VisionLANLoss"]


class CTCLoss(LossBase):
    """
    CTCLoss definition

    Args:
        pred_seq_len(int): the length of the predicted character sequence. For text images, this value equals to
            W - the width of feature map encoded by the visual bacbkone.
            This can be obtained by probing the output shape in the network.
            E.g., for a training image in shape (3, 32, 100), the feature map encoded by resnet34 bacbkone is
            in shape (512, 1, 4), W = 4, sequence len is 4.
        max_label_len(int): the maximum number of characters in a text label, i.e. max_text_len in yaml.
        batch_size(int): batch size of input logits. bs
    """

    def __init__(
        self, pred_seq_len: int = 26, max_label_len: int = 25, batch_size: int = 32, reduction: str = "mean"
    ) -> None:
        super(CTCLoss, self).__init__(reduction=reduction)
        assert pred_seq_len > max_label_len, (
            "pred_seq_len is required to be larger than max_label_len for CTCLoss. Please adjust the strides in the "
            "backbone, or reduce max_text_length in yaml"
        )
        self.sequence_length = Tensor(np.array([pred_seq_len] * batch_size), ms.int32)

        label_indices = []
        for i in range(batch_size):
            for j in range(max_label_len):
                label_indices.append([i, j])
        self.label_indices = Tensor(np.array(label_indices), ms.int64)
        self.ctc_loss = ops.CTCLoss(ctc_merge_repeated=True)

    def construct(self, pred: Tensor, label: Tensor) -> Tensor:
        """
        Args:
            pred (Tensor): network prediction which is a
                logit Tensor in shape (W, BS, NC), where W - seq len, BS - batch size. NC - num of classes
                (types of character + blank + 1)
            label (Tensor): GT sequence of character indices in shape (BS, SL), SL - sequence length, which is padded to
                max_text_length
        Returns:
            loss value (Tensor)
        """
        logit = pred
        label_values = ops.reshape(label, (-1,))

        loss, _ = self.ctc_loss(logit, self.label_indices, label_values, self.sequence_length)
        loss = self.get_loss(loss)
        return loss


class VisionLANLoss(LossBase):
    """VisionLAN Loss. It predicts the cross entropy loss while ignoring the target value\
        that equals to -100.
    Args:
        mode (str): mode of the loss, selected from ["LF_1", "LF_2", "LA"]. Default: "LF_1".
        weight_res (float): weight of the remaining text prediction loss. Default: 0.5.
        weight_mas (float): weight of the masked text prediction loss. Default: 0.5.
        reduction (str): reduction method. Default: "mean".
    """

    def __init__(self, mode="LF_1", weight_res=0.5, weight_mas=0.5, reduction="mean", **kwargs):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            reduction=reduction, ignore_index=-100
        )  # ignore the samples in the target padded with -100
        assert mode in ["LF_1", "LF_2", "LA"]
        self.mode = mode
        self.weight_res = weight_res
        self.weight_mas = weight_mas

    def replace_label_with_target_value(self, target, label_length, target_value=-100):
        """In each row of target, replace the elements (zeros) by the pad value.
        Args:
            target: (Tensor), target text indexes, shape (B, max_len)
            target_value: (int), the value used to replace the padded label. Default: -100.

        Returns:
            target: (Tensor), target text indexes, shape (B, max_len)
        """
        b, max_len = target.shape
        # label_length is the length of valid characters
        # 1. update the invalid characters except for the first invalid character to the target value
        # 2. if some samples' lengths equal to max_len, then the tensor would not be updated
        indices = label_length[:, None]
        nonzero_mask = ops.cast(target != 0, ms.float32)
        updates = ops.ones(indices.shape, nonzero_mask.dtype)
        nonzero_mask = ops.tensor_scatter_elements(nonzero_mask, indices, updates, axis=1)
        nonzero_mask = ops.cast(nonzero_mask, ms.bool_)
        target[~nonzero_mask] = target_value
        return target

    def construct(self, predicts, label, label_res, label_sub, label_length):
        text_pre = predicts[0]
        b, l, c = text_pre.shape
        target = ops.cast(label, ms.int32)  # target text indexes
        label_length = ops.cast(label_length, ms.int32)
        target = self.replace_label_with_target_value(target, label_length)
        if self.mode == "LF_1":  # train the backbone, sequence model, and prediction layer
            loss = self.criterion(
                text_pre.view(b * l, c),
                target.view(
                    b * l,
                ),
            )
        else:  # train the backbone, sequence model, and prediction layer with masking
            text_rem = predicts[1]
            b1, l1, c1 = text_rem.shape
            text_mas = predicts[2]
            b2, l2, c2 = text_mas.shape
            target_res = ops.cast(label_res, ms.int32)
            target_sub = ops.cast(label_sub, ms.int32)
            target_res = self.replace_label_with_target_value(target_res, label_length - 1)
            target_sub = self.replace_label_with_target_value(target_sub, ops.ones((len(label_length),), ms.int32))
            loss_ori = self.criterion(
                text_pre.view(b * l, c),
                target.view(
                    b * l,
                ),
            )
            loss_res = self.criterion(
                text_rem.view(b1 * l1, c1),
                target_res.view(
                    b1 * l1,
                ),
            )
            loss_mas = self.criterion(
                text_mas.view(b2 * l2, c2),
                target_sub.view(
                    b2 * l2,
                ),
            )
            loss = loss_ori + loss_res * self.weight_res + loss_mas * self.weight_mas
        return loss


class AttentionLoss(LossBase):
    def __init__(self, reduction: str = "mean", ignore_index: int = 0) -> None:
        super().__init__()
        # ignore <GO> symbol, assume it is placed at 0th index
        self.criterion = nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)

    def construct(self, logits: Tensor, labels: Tensor) -> Tensor:
        labels = labels[:, 1:]  # wihout <GO> symbol
        num_classes = logits.shape[-1]
        logits = ops.reshape(logits, (-1, num_classes))
        labels = ops.reshape(labels, (-1,))
        return self.criterion(logits, labels)
