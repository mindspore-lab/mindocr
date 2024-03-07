from functools import wraps

import numpy as np
import mindspore as ms
from mindspore import Tensor

from numpy.random import Generator, PCG64
from mindocr.models.utils.deepsolo.deepsolo_layers import *
from mindocr.models.utils.deepsolo.deformable_transformer import *

from inspect import isfunction

from mindspore.nn import Dense

import yaml

from test_utils import dense_weight_generator, dense_bias_generator

class Dense_mock(Dense):
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init=None,
                 bias_init=None,
                 has_bias=True,
                 activation=None,
                 dtype=mstype.float32):
        w = dense_weight_generator(42, [out_channels, in_channels], ms.float32)[0]
        b = dense_bias_generator(42, out_channels, ms.float32)[0]
        super(Dense_mock, self).__init__(
            in_channels,
            out_channels,
            weight_init=w,
            bias_init=b
        )