import mindspore as ms
from mindspore import nn

__all__ = ["VQASerTokenLayoutLMLoss", "VQAReTokenLayoutLMLoss"]


class VQASerTokenLayoutLMLoss(nn.LossBase):
    """
    Loss for token classification task.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fct = nn.CrossEntropyLoss()

    def construct(self, predicts, attention_mask, labels):
        loss = self.loss_fct(predicts.transpose(0, 2, 1), labels.astype(ms.int32))
        return loss


class VQAReTokenLayoutLMLoss(nn.LossBase):

    """
    Loss for relation extraction task.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fct = nn.CrossEntropyLoss()

    def construct(self, predicts, attention_mask, labels):
        loss = self.loss_fct(predicts.transpose(0, 2, 1), labels.astype(ms.int32))
        return loss
