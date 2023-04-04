import numpy as np

import mindspore as ms
from mindspore import ops
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn.optim import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register


def _check_param_value(beta1, beta2, prim_name):
    """Check the type of inputs."""
    assert isinstance(beta1, float), f"For '{prim_name}', the type of 'beta1' must be 'float', but got type '{type(beta1).__name__}'."
    assert isinstance(beta2, float), f"For '{prim_name}', the type of 'beta2' must be 'float', but got type '{type(beta2).__name__}'."
    assert 0.0 < beta1 < 1.0, f"For '{prim_name}', the range of 'beta1' must be (0.0, 1.0), but got {beta1}."
    assert 0.0 < beta2 < 1.0, f"For '{prim_name}', the range of 'beta2' must be (0.0, 1.0), but got {beta2}."


_grad_scale = ops.MultitypeFuncGraph("grad_scale")
map_ = ops.Map()


@_grad_scale.register("Number", "Tensor")
def tensor_grad_scale(scale, grad):
    """Get grad with scale."""
    if scale == 1.0:
        return grad
    return ops.mul(grad, ops.cast(scale, grad.dtype))


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale_with_tensor(scale, grad):
    """Get grad with scale."""
    return ops.mul(grad, ops.cast(scale, grad.dtype))


def scale_grad(gradients, reciprocal_scale):
    gradients = map_(ops.partial(_grad_scale, reciprocal_scale), gradients)
    return gradients


_lion_opt = ops.MultitypeFuncGraph("lion_opt")
_scaler_one = Tensor(1, ms.int32)


@_lion_opt.register(
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Bool",
    "Bool",
)
def _update_run_op(
    beta1_power,
    beta2_power,
    beta1,
    beta2,
    lr,
    weight_decay,
    param,
    m,
    gradient,
    decay_flag,
    optim_filter,
):
    """
    Update parameters.
    Args:
        beta1 (Tensor): The exponential decay rate for the 1st moment estimations. Should be in range (0.0, 1.0).
        beta2 (Tensor): The exponential decay rate for the 2nd moment estimations. Should be in range (0.0, 1.0).
        lr (Tensor): Learning rate.
        weight_decay (Tensor): Weight decay. Should be equal to or greater than 0.
        param (Tensor): Parameters.
        m (Tensor): m value of parameters.
        gradient (Tensor): Gradient of parameters.
        decay_flag (bool): Applies weight decay or not.
        optim_filter (bool): Applies parameter update or not.
    Returns:
        Tensor, the new value of v after updating.
    """
    if optim_filter:
        param_fp32 = ops.cast(param, ms.float32)
        m_fp32 = ops.cast(m, ms.float32)
        gradient_fp32 = ops.cast(gradient, ms.float32)

        c_t = ops.mul(beta1, m_fp32) + ops.mul(ops.cast(ops.tuple_to_array((1.0,)), ms.float32) - beta1, gradient_fp32)

        update = ops.Sign()(c_t)

        if decay_flag:
            update = ops.mul(weight_decay, param_fp32) + update

        update_with_lr = ops.mul(lr, update)
        next_param = param_fp32 - ops.reshape(update_with_lr, ops.shape(param_fp32))

        next_m = ops.mul(beta2, m_fp32) + ops.mul(
            ops.cast(ops.tuple_to_array((1.0,)), ms.float32) - beta2, gradient_fp32
        )

        next_param = ops.depend(next_param, ops.assign(param, ops.cast(next_param, param.dtype)))
        next_param = ops.depend(next_param, ops.assign(m, ops.cast(next_m, m.dtype)))

        return ops.cast(next_param, param.dtype)
    return gradient


class Lion(Optimizer):
    """
    Implementation of Lion optimizer from paper 'https://arxiv.org/abs/2302.06675'.
    Additionally, this implementation is with gradient clipping.

    Notes:
    lr is usually 3-10x smaller than adamw.
    weight decay is usually 3-10x larger than adamw.
    """

    @opt_init_args_register
    def __init__(
        self,
        params,
        learning_rate=2e-4,
        beta1=0.9,
        beta2=0.99,
        weight_decay=0.0,
        loss_scale=1.0,
        clip=False,
    ):
        super().__init__(learning_rate, params, weight_decay)
        _check_param_value(beta1, beta2, self.cls_name)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.moments1 = self.parameters.clone(prefix="lion_m", init="zeros")
        self.hyper_map = ops.HyperMap()
        self.beta1_power = Parameter(initializer(1, [1], ms.float32), name="beta1_power")
        self.beta2_power = Parameter(initializer(1, [1], ms.float32), name="beta2_power")

        self.reciprocal_scale = Tensor(1.0 / loss_scale, ms.float32)
        self.clip = clip

    def construct(self, gradients):
        lr = self.get_lr()
        gradients = scale_grad(gradients, self.reciprocal_scale)
        if self.clip:
            gradients = ops.clip_by_global_norm(gradients, 5.0, None)

        beta1_power = self.beta1_power * self.beta1
        self.beta1_power = beta1_power
        beta2_power = self.beta2_power * self.beta2
        self.beta2_power = beta2_power

        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(
                    ops.partial(_lion_opt, beta1_power, beta2_power, self.beta1, self.beta2),
                    lr,
                    self.weight_decay,
                    self.parameters,
                    self.moments1,
                    gradients,
                    self.decay_flags,
                    self.optim_filter,
                )
            else:
                optim_result = self.hyper_map(
                    ops.partial(_lion_opt, beta1_power, beta2_power, self.beta1, self.beta2, lr),
                    self.weight_decay,
                    self.parameters,
                    self.moments1,
                    gradients,
                    self.decay_flags,
                    self.optim_filter,
                )
        else:
            optim_result = self.hyper_map(
                ops.partial(_lion_opt, beta1_power, beta2_power, self.beta1, self.beta2, lr, self.weight_decay),
                self.parameters,
                self.moments1,
                gradients,
                self.decay_flags,
                self.optim_filter,
            )
        if self.use_parallel:
            self.broadcast_params(optim_result)
        return optim_result
