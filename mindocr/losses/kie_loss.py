import mindspore as ms
from mindspore import nn, ops

__all__ = ["VQASerTokenLayoutLMLoss", "LossFromOutput"]


class VQASerTokenLayoutLMLoss(nn.LossBase):
    """
    Loss for token classification task.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.loss_class = nn.CrossEntropyLoss()

    def construct(self, predicts, attention_mask, labels):
        loss = self.loss_fct(predicts.transpose(0, 2, 1), labels.astype(ms.int32))
        return loss


class LossFromOutput(nn.LossBase):

    """
    Get loss from network output
    """

    def __init__(self, key="loss", reduction="none"):
        super().__init__()
        self.key = key
        self.reduction = reduction

    def construct(self, predicts, entities, relations):
        loss = predicts
        if self.key is not None and isinstance(predicts, dict):
            loss = loss[self.key]
        if self.reduction == "mean":
            loss = ops.mean(loss)
        elif self.reduction == "sum":
            loss = ops.sum(loss)
        return loss
