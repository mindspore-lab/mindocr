from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mindspore as ms
from mindspore import nn
from mindspore.nn.loss.loss import LossBase
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore import ops 


class CTCLoss(nn.Cell):
    def __init__(self, use_focal_loss=False, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')
        self.use_focal_loss = use_focal_loss

    def forward(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        predicts = predicts.transpose((1, 0, 2))
        N, B, _ = predicts.shape
        preds_lengths = paddle.to_tensor(
            [N] * B, dtype='int64', place=paddle.CPUPlace())
        labels = batch[1].astype("int32")
        label_lengths = batch[2].astype('int64')
        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        if self.use_focal_loss:
            weight = paddle.exp(-loss)
            weight = paddle.subtract(paddle.to_tensor([1.0]), weight)
            weight = paddle.square(weight)
            loss = paddle.multiply(loss, weight)
        loss = loss.mean()
        return {'loss': loss}

class CTCLoss(LossBase):
    """
     CTCLoss definition

     Args:
        max_sequence_length(int): max number of sequence length. For text images, the value is equal to image
        width
        max_label_length(int): max number of label length for each input.
        batch_size(int): batch size of input logits
     """

    def __init__(self, max_sequence_length, max_label_length, batch_size):
        super(CTCLoss, self).__init__()
        self.sequence_length =  
        labels_indices = []
        for i in range(batch_size):
            for j in range(max_label_length):
                labels_indices.append([i, j])
        self.labels_indices = Parameter(Tensor(np.array(labels_indices), mstype.int64), name="labels_indices")
        #self.reshape = P.Reshape()
        self.ctc_loss = ops.CTCLoss(ctc_merge_repeated=True)

    def construct(self, pred, label):
        '''
        Args:
            pred (dict): {head_out: logits}
                        logits is a Tensor in shape (W, BS, num_classes), where W - seq len, BS - batch size.
            label (Tensor): GT sequence of character indices in shape (BS, max_text_length)
        Returns:
            loss value
        '''
        logit = pred['head_out']
        labels_values = ops.reshape(label, (-1,))
        loss, _ = self.ctc_loss(logit, self.labels_indices, labels_values, self.sequence_length)
        return loss
