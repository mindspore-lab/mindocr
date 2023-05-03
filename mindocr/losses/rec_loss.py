import mindspore as ms
from mindspore import Tensor
from mindspore import nn
from mindspore.nn.loss.loss import LossBase
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore import ops
import numpy as np

__all__ = ['CTCLoss']

# TODO: support label_weights for imbalance data
class CTCLoss(LossBase):
    """
     CTCLoss definition

     Args:
        pred_seq_len(int): the length of the predicted character sequence. For text images, this value equals to W - the width of feature map encoded by the visual bacbkone. This can be obtained by probing the output shape in the network.
            E.g., for a training image in shape (3, 32, 100), the feature map encoded by resnet34 bacbkone is in shape (512, 1, 4), W = 4, sequence len is 4.
        max_label_len(int): the maximum number of characters in a text label, i.e. max_text_len in yaml.
        batch_size(int): batch size of input logits. bs
     """

    def __init__(self, pred_seq_len=26, max_label_len=25, batch_size=32, reduction='mean'):
        super(CTCLoss, self).__init__()
        assert pred_seq_len > max_label_len, 'pred_seq_len is required to be larger than max_label_len for CTCLoss. Please adjust the strides in the backbone, or reduce max_text_length in yaml'
        self.sequence_length = Tensor(np.array([pred_seq_len] * batch_size), mstype.int32)
        label_indices = []
        for i in range(batch_size):
            for j in range(max_label_len):
                label_indices.append([i, j])
        self.label_indices = Tensor(np.array(label_indices), mstype.int64)
        #self.reshape = P.Reshape()
        self.ctc_loss = ops.CTCLoss(ctc_merge_repeated=True)

        self.reduction = reduction
        print('D: ', self.label_indices.shape)

    # TODO: diff from paddle, paddle takes `label_length` as input too.
    def construct(self, pred: Tensor, label: Tensor):
        '''
        Args:
            pred (Tensor): network prediction which is a
                logit Tensor in shape (W, BS, NC), where W - seq len, BS - batch size. NC - num of classes (types of character + blank + 1)
            label (Tensor): GT sequence of character indices in shape (BS, SL), SL - sequence length, which is padded to max_text_length
        Returns:
            loss value (Tensor)
        '''
        logit = pred
        #T, bs, nc = logit.shape
        #logit = ops.reshape(logit, (T*bs, nc))
        label_values = ops.reshape(label, (-1,))

        loss, _ = self.ctc_loss(logit, self.label_indices, label_values, self.sequence_length)

        if self.reduction=='mean':
            loss = loss.mean()

        return loss


if __name__ == '__main__':
    max_text_length = 23
    nc = 26
    bs = 32
    pred_seq_len  = 24

    loss_fn = CTCLoss(pred_seq_len, max_text_length, bs)

    x = ms.Tensor(np.random.rand(pred_seq_len, bs, nc), dtype=ms.float32)
    label = ms.Tensor(np.random.randint(0, nc,  size=(bs, max_text_length)), dtype=ms.int32)

    loss = loss_fn(x, label)
    print(loss)
