import mindspore as ms
from mindspore import nn, ops

__all__ = ["VQASerTokenLayoutLMLoss", "LossFromOutput"]


class VQASerTokenLayoutLMLoss(nn.LossBase):
    """
    Loss for token classification task.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.loss_class = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="none")
        self.num_classes = num_classes

    def construct(self, predicts, attention_mask, labels):
        if attention_mask is not None:
            loss = self.loss_class(
                predicts.reshape((-1, self.num_classes)).astype(ms.float32),
                labels.reshape((-1,)).astype(ms.int32),
            )
            attention_mask = attention_mask.reshape((-1,))
            loss = ops.mul(loss, attention_mask)
            loss = loss[loss > 0]
        else:
            loss = self.loss_class(
                predicts.reshape((-1, self.num_classes)).astype(ms.float32),
                labels.reshape((-1,)).astype(ms.int32),
            )
        return ops.reduce_mean(loss)


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
