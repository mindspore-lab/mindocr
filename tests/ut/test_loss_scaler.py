import sys
sys.path.append('.')

import numpy as np
import mindspore
from mindspore import Tensor, Parameter, nn
import mindspore.ops as ops
import pytest
from mindocr.utils.loss_scaler import get_loss_scales
from addict import Dict

class Net(nn.Cell):
    def __init__(self, in_features, out_features):
        super(Net, self).__init__()
        self.weight = Parameter(Tensor(np.ones([in_features, out_features]).astype(np.float32)),
                                name='weight')
        self.matmul = ops.MatMul()

    def construct(self, x):
        output = self.matmul(x, self.weight)
        return output
 
static_cfg = {} 
static_cfg['loss_scaler'] = {'type': 'static', 
                            'loss_scale': 1.0}
static_cfg = Dict(static_cfg)

dynamic_cfg = {}
dynamic_cfg['loss_scaler'] = { 'type': 'dynamic', 
                            'loss_scale': 1024.0,
                            'scale_factor': 2.0,
                            'scale_window': 2}
dynamic_cfg = Dict(dynamic_cfg) 


@pytest.mark.parametrize('ls_type', ['static', 'dynamic'])
@pytest.mark.parametrize('drop_overflow_update', [True, False])
def test_loss_scaler(ls_type, drop_overflow_update):
    in_features, out_features = 16, 10
    net = Net(in_features, out_features)
    loss = nn.MSELoss()
    net_with_loss = nn.WithLossCell(net, loss)

    #manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=1000)
    if ls_type == 'static':
        cfg = static_cfg
    elif ls_type == 'dynamic':
        cfg = dynamic_cfg
    cfg.system.drop_overflow_update = drop_overflow_update

    manager, opt_loss_scale = get_loss_scales(cfg)
    
    optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9, loss_scale=opt_loss_scale)
    train_network = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=manager)

    input = Tensor(np.ones([out_features, in_features]), mindspore.float32)
    labels = Tensor(np.ones([out_features,]), mindspore.float32)
    
    loss_scales = []
    for i in range(3):
        loss, is_overflow, loss_scale_updated = train_network(input, labels)
        loss_scales.append(float(loss_scale_updated.asnumpy()))
        print(loss)

    print(loss_scales) 

    if ls_type == 'static': 
        assert loss_scales[0] == loss_scales[-1]
    elif ls_type == 'dynamic':
        assert loss_scales[0] != loss_scales[-1]

if __name__ == '__main__':
    test_loss_scaler('static', True)
