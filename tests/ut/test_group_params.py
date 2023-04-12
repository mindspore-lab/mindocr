import sys
sys.path.append('.')

import pytest
import numpy as np
import mindspore as ms
import mindocr
from mindocr.models import build_model
from mindocr.optim import create_optimizer


def param_grouping_svtr(params, weight_decay):
    decay_params = []
    no_decay_params = []

    filter_keys = ['beta', 'gamma', 'pos_emb'] # correspond to nn.BatchNorm, nn.LayerNorm, and position embedding layer if named as 'pos_emb'. TODO: check svtr naming.

    for param in params:
        filter_param = False
        for k in filter_keys:
            if k in param.name:
                filter_param = True

        if filter_param:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params},
        {"order_params": params},
    ]


def build_group_params(params, weight_decay, strategy=None, no_weight_decay=['bias', 'beta', 'gamma']):
    if strategy is not None:
        if strategy == 'svtr':
            return param_grouping_svtr(params, weight_decay)
    else:
        for param in params:
            filter_param = False
            for k in no_weight_decay:
                if k in param.name:
                    filter_param = True

            if filter_param:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params},
            {"order_params": params},
        ]


def test_group_params(model_name):
    network = build_model(model_name)
    WD = 0.1

    params = build_group_params(network.trainable_params(), WD, no_weight_decay=['beta', 'gamma', 'pos_emb'])

    optimizer = create_optimizer(
            params,
            'momentum',
            lr=0.01,
            weight_decay=WD,
            momentum=0.9,
            filter_bias_and_bn=False,
        )

    # TODO: test training
