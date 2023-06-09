import numpy as np

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.nn.loss.loss import LossBase

__all__ = ["CTCLoss", "AttentionLoss"]


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


class AttentionLoss(LossBase):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        # ignore <GO> symbol, assume it is placed at 0th index
        self.criterion = nn.CrossEntropyLoss(reduction=reduction, ignore_index=0)

    def construct(self, logits: Tensor, labels: Tensor) -> Tensor:
        labels = labels[:, 1:]  # wihout <GO> symbol
        num_classes = logits.shape[-1]
        logits = ops.reshape(logits, (-1, num_classes))
        labels = ops.reshape(labels, (-1,))
        return self.criterion(logits, labels)
