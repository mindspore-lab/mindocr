# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""adan"""
from __future__ import absolute_import

from mindspore.common import dtype as mstype
from mindspore.common.api import ms_function
from mindspore.common.tensor import Tensor
from mindspore.nn.optim.optimizer import Optimizer, opt_init_args_register
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

_adan_opt = C.MultitypeFuncGraph("adan_opt")
_scaler_one = Tensor(1, mstype.int32)
_scaler_ten = Tensor(10, mstype.float32)


@_adan_opt.register(
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
)
def _update_run_op(
    beta1,
    beta2,
    beta3,
    eps,
    lr,
    weight_decay,
    param,
    m,
    v,
    n,
    gradient,
    prev_gradient,
):
    """
    Update parameters.

    Args:
        beta1 (Tensor): The exponential decay rate for the 1st moment estimations. Should be in range (0.0, 1.0).
        beta2 (Tensor): The exponential decay rate for the 2nd moment estimations. Should be in range (0.0, 1.0).
        eps (Tensor): Term added to the denominator to improve numerical stability. Should be greater than 0.
        lr (Tensor): Learning rate.
        weight_decay (numbers.Number): Weight decay. Should be equal to or greater than 0. if 0, no decay
        param (Tensor): Parameters.
        m (Tensor): m value of parameters.
        v (Tensor): v value of parameters.
        gradient (Tensor): Gradient of parameters.

    Returns:
        Tensor, the new value of v after updating.
    """
    op_cast = P.Cast()
    op_mul = P.Mul()
    op_square = P.Square()
    op_sqrt = P.Sqrt()
    op_cast = P.Cast()
    op_reshape = P.Reshape()
    op_shape = P.Shape()

    success = True

    # if global_step == 0.0: # init
    # TODO: use global_step==0 as the condition to init prev_gradient as gradient
    # if (F.reduce_min(prev_gradient) == 0.0) and (F.reduce_max(prev_gradient) == 0.0):
    if F.reduce_sum(prev_gradient) == 0.0:
        success = F.depend(success, F.assign(prev_gradient, gradient))

    # TODO: is casting needed?
    param_fp32 = op_cast(param, mstype.float32)
    m_fp32 = op_cast(m, mstype.float32)
    v_fp32 = op_cast(v, mstype.float32)
    n_fp32 = op_cast(n, mstype.float32)
    gradient_fp32 = op_cast(gradient, mstype.float32)
    prev_gradient_fp32 = op_cast(prev_gradient, mstype.float32)

    next_m = op_mul(F.tuple_to_array((1.0,)) - beta1, m_fp32) + op_mul(beta1, gradient_fp32)

    next_v = op_mul(F.tuple_to_array((1.0,)) - beta2, v_fp32) + op_mul(beta2, gradient_fp32 - prev_gradient_fp32)

    next_n = op_mul(F.tuple_to_array((1.0,)) - beta3, n_fp32) + op_mul(
        beta3, op_square(gradient + op_mul(F.tuple_to_array((1.0,)) - beta2, gradient_fp32 - prev_gradient_fp32))
    )

    lr_t = lr / (eps + op_sqrt(next_n))

    update = next_m + op_mul(F.tuple_to_array((1.0,)) - beta2, next_v)

    # if decay_flag:
    #    update = op_mul(weight_decay, param_fp32) + update

    next_param = param_fp32 - op_reshape(op_mul(lr_t, update), op_shape(param_fp32))

    next_param = next_param / (Tensor(1.0, mstype.float32) + op_mul(weight_decay, lr_t))

    success = F.depend(success, F.assign(param, op_cast(next_param, F.dtype(param))))
    success = F.depend(success, F.assign(m, op_cast(next_m, F.dtype(m))))
    success = F.depend(success, F.assign(v, op_cast(next_v, F.dtype(v))))
    success = F.depend(success, F.assign(n, op_cast(next_n, F.dtype(n))))
    success = F.depend(success, F.assign(prev_gradient, gradient))

    return op_cast(next_param, F.dtype(param))


def _check_param_value(beta1, beta2, eps, use_locking, prim_name):
    """Check the type of inputs."""
    assert isinstance(beta1, float), f"For '{prim_name}', the type of 'beta1' must be 'float', but got type '{type(beta1).__name__}'."
    assert isinstance(beta2, float), f"For '{prim_name}', the type of 'beta2' must be 'float', but got type '{type(beta2).__name__}'."
    assert isinstance(eps, float), f"For '{prim_name}', the type of 'eps' must be 'float', but got type '{type(eps).__name__}'."
    assert 0.0 < beta1 < 1.0, f"For '{prim_name}', the range of 'beta1' must be (0.0, 1.0), but got {beta1}."
    assert 0.0 < beta2 < 1.0, f"For '{prim_name}', the range of 'beta2' must be (0.0, 1.0), but got {beta2}."
    assert eps > 0, f"For '{prim_name}', the 'eps' must be positive, but got {eps}."
    assert isinstance(use_locking, bool), f"For '{prim_name}', the type of 'use_locking' must be 'bool', but got type '{type(use_locking).__name__}'."


class Adan(Optimizer):
    """
    The Adan (ADAptive Nesterov momentum algorithm) Optimizer from https://arxiv.org/abs/2208.06677

    Note: it is an experimental version.
    """

    @opt_init_args_register
    def __init__(
        self,
        params,
        learning_rate=1e-3,
        beta1=0.98,
        beta2=0.92,
        beta3=0.99,
        eps=1e-8,
        use_locking=False,
        weight_decay=1e-6,
        loss_scale=1.0,
    ):
        super().__init__(
            learning_rate, params, weight_decay=weight_decay, loss_scale=loss_scale
        )  # Optimized inherit weight decay is bloaked. weight decay is computed in this py.

        _check_param_value(beta1, beta2, eps, use_locking, self.cls_name)

        self.beta1 = Tensor(beta1, mstype.float32)
        self.beta2 = Tensor(beta2, mstype.float32)
        self.beta3 = Tensor(beta3, mstype.float32)
        # self.beta1_power = Parameter(initializer(1, [1], mstype.float32), name="beta1_power")
        # self.beta2_power = Parameter(initializer(1, [1], mstype.float32), name="beta2_power")
        # self.beta3_power = Parameter(initializer(1, [1], mstype.float32), name="beta3_power")
        self.eps = Tensor(eps, mstype.float32)
        self.use_locking = use_locking
        self.moment1 = self._parameters.clone(prefix="moment1", init="zeros")  # m
        self.moment2 = self._parameters.clone(prefix="moment2", init="zeros")  # v
        self.moment3 = self._parameters.clone(prefix="moment3", init="zeros")  # n
        self.prev_gradient = self._parameters.clone(prefix="prev_gradient", init="zeros")
        # print('prev g: ', type(self.prev_gradient))

        self.weight_decay = Tensor(weight_decay, mstype.float32)

    @ms_function
    def construct(self, gradients):
        params = self._parameters
        moment1 = self.moment1
        moment2 = self.moment2
        moment3 = self.moment3
        # vhat = self.vhat
        gradients = self.flatten_gradients(gradients)
        # gradients = self.decay_weight(gradients) # we decay weight in adan_opt func
        gradients = self.gradients_centralization(gradients)
        gradients = self.scale_grad(gradients)
        gradients = self._grad_sparse_indices_deduplicate(gradients)
        lr = self.get_lr()
        # weight_decay = self.get_weight_decay()

        # if self.global_step == 0:
        #    success = F.depend(True, F.assign(self.prev_gradient, gradients))

        # TODO: currently not support dist
        success = self.map_(
            F.partial(_adan_opt, self.beta1, self.beta2, self.beta3, self.eps, lr, self.weight_decay),
            params,
            moment1,
            moment2,
            moment3,
            gradients,
            self.prev_gradient,
        )
        # params, moment1, moment2, moment3, gradients, self.prev_gradient, self.global_step)

        return success

    @Optimizer.target.setter
    def target(self, value):
        """
        If the input value is set to "CPU", the parameters will be updated on the host using the Fused
        optimizer operation.
        """
        self._set_base_target(value)
