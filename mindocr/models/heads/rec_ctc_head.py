import mindspore as ms
from mindspore import nn
from mindspore import ops

__all__ = ['CTCHead']

class CTCHead(nn.Cell):
    '''
    An MLP module for CTC Loss.
    For CRNN, the input should be in shape [W, BS, 2*C], which is output by RNNEncoder.
    The MLP encodes and classifies the features, then return a logit tensor in shape [W, BS, num_classes]
    For chinese words, num_classes can be over 60,000, so weight regulaization may matter.

    Args:

    Example:

    '''
    # TODO: add dropout regularization. I think it will benefit the performance of 2-layer MLP for chinese text recoginition.
    def __init__(self,
                 in_channels,
                 out_channels,
                 #fc_decay: float=0.0004,
                 mid_channels: int=None,
                 return_feats: bool=False,
                 weight_init: str='normal', #'xavier_uniform',
                 bias_init: str='zeros', #'xavier_uniform',
                 dropout: float=0.):
        super().__init__()
        # TODO:Diff: 1. paddle initialize weight and bias with a Xaivier Uniform variant.   2. paddle uses L2 decay on FC weight and bias with specified decay factor fc_decay 0.00002.

        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats

        # TODO: paddle is not using the exact XavierUniform. It uses check which is better.
        #w_init = 'xavier_uniform'
        #b_init = 'xavier_uniform'
        if mid_channels is None:
            self.dense1 = nn.Dense(in_channels, out_channels, weight_init=weight_init, bias_init=bias_init)
        else:
            # TODO: paddle did not use activation after linear, why no activation?
            self.dense1 = nn.Dense(in_channels, mid_channels, weight_init=weight_init, bias_init=bias_init)
            #self.activation = nn.GeLU()
            self.dropout = nn.Dropout(keep_prob=1-dropout)
            self.dense2 = nn.Dense(mid_channels, out_channels, weight_init=weight_init, bias_init=bias_init)
            #self.dropout = nn.Dropout(keep_prob)

    def construct(self, x):
        """Feed Forward construct.
        Args:
            x (Tensor): feature in shape [W, BS, 2*C]
        Returns:
            h (Tensor): logits in shape [W, BS, num_classes], where W - sequence len, fixed.
        """
        h = self.dense1(x)
        #x = self.dropout(x)
        if self.mid_channels is not None:
            h = self.dropout(h)
            h = self.dense2(h)

        if not self.training:
            h = ops.softmax(h, axis=2)

        pred = {'head_out': h}
        return pred


if __name__ == '__main__':
    import numpy as np
    #from mindocr.utils.debug import initialize_network_with_constant
    w, bs, c = 16, 8, 256
    x = ms.Tensor(np.random.rand(w, bs, c), dtype=ms.float32)

    model = CTCHead(in_channels=256, out_channels=26+3, mid_channels=128)
    #initialize_network_with_constant(model2, c_weight=1.0)
    y = model(x)

    print('output', y.shape, y.sum())
