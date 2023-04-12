import sys
sys.path.append('.')

import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor
import mindocr
from mindocr.models import build_model
from mindocr import build_loss
from mindocr.optim import create_optimizer, create_group_params
from mindspore.nn import TrainOneStepCell, WithLossCell


@pytest.mark.parametrize("strategy", [None, 'svtr', 'filter_norm_and_bias'])
@pytest.mark.parametrize("nwd_params", [['beta', 'gamma', 'bias']])
def test_group_params(strategy, nwd_params):
    network = build_model('crnn_r34')
    WD = 1e-5

    params = create_group_params(network.trainable_params(), WD, strategy, no_weight_decay_params=nwd_params)

    assert 'weight_decay' in params[0]

    optimizer = create_optimizer(
            params,
            'momentum',
            lr=1e-5,
            weight_decay=WD,
            momentum=0.9,
            filter_bias_and_bn=False,
        )
    bs = 8
    max_ll = 24
    loss_cfg = {"pred_seq_len": 25, "max_label_len": max_ll, "batch_size": bs}
    loss_fn = build_loss('CTCLoss', **loss_cfg)

    # TODO: test training

    input_data = Tensor(np.ones([bs, 3, 32, 100]).astype(np.float32) * 0.1)
    label = Tensor(np.ones([bs, max_ll]).astype(np.int32))

    net_with_loss = WithLossCell(network, loss_fn)
    train_network = TrainOneStepCell(net_with_loss, optimizer)

    train_network.set_train()

    begin_loss = train_network(input_data, label)
    for i in range(2):
        cur_loss = train_network(input_data, label)

if __name__ == '__main__':
    test_group_params(None, ['bias'])
