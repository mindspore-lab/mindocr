from mindspore import nn, ops

__all__ = ["VQASerTokenLayoutLMLoss", "LossFromOutput"]


class VQASerTokenLayoutLMLoss(nn.LossBase):
    def __init__(self, num_classes, key=None):
        super().__init__()
        self.loss_class = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.ignore_index = self.loss_class.ignore_index
        self.key = key

    def construct(self, predicts, batch):
        if isinstance(predicts, dict) and self.key is not None:
            predicts = predicts[self.key]
        labels = batch[5]
        attention_mask = batch[2]
        if attention_mask is not None:
            active_loss = attention_mask.reshape((-1,)) == 1
            active_output = predicts.reshape((-1, self.num_classes))[active_loss]
            active_label = labels.reshape((-1,))[active_loss]
            loss = self.loss_class(active_output, active_label)
        else:
            loss = self.loss_class(predicts.reshape((-1, self.num_classes)), labels.reshape((-1,)))
        return loss


class LossFromOutput(nn.LossBase):
    def __init__(self, key="loss", reduction="none"):
        super().__init__()
        self.key = key
        self.reduction = reduction

    def construct(self, predicts, batch):
        loss = predicts
        if self.key is not None and isinstance(predicts, dict):
            loss = loss[self.key]
        if self.reduction == "mean":
            loss = ops.mean(loss)
        elif self.reduction == "sum":
            loss = ops.sum(loss)
        return loss
