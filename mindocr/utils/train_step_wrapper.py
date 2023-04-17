"""Train step wrapper supporting setting drop overflow update, ema etc"""
import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common import RowTensor
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
import mindspore.context as context


_ema_op = C.MultitypeFuncGraph("grad_ema_op")
_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()
_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")


@_ema_op.register("Tensor", "Tensor", "Tensor")
def _ema_weights(factor, ema_weight, weight):
    return F.assign(ema_weight, ema_weight * factor + weight * (1 - factor))


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(
        grad.indices,
        grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
        grad.dense_shape,
    )

class TrainOneStepWrapper(nn.TrainOneStepWithLossScaleCell):
    """TrainStep with ema and clip grad.
    Args:
        drop_overflow_update: if True, network will not be updated when gradient is overflow.
        scale_sense (Union[Tensor, Cell]): If this value is a Cell, it will be called
            to update loss scale. If this value is a Tensor, the loss scale can be modified by `set_sense_scale`,
            the shape should be :math:`()` or :math:`(1,)`.

    """

    def __init__(
        self,
        network,
        optimizer,
        scale_sense=1.0,
        ema=False,
        ema_decay=0.9999,
        updates=0,
        drop_overflow_update=True,
        gradient_accumulation_steps=1,
        clip_grad=False,
        clip_norm=1.0,
        verbose=False,
    ):
        super().__init__(network, optimizer, scale_sense)
        self.ema = ema
        self.ema_decay = ema_decay
        self.updates = Parameter(Tensor(updates, ms.float32), requires_grad=False)
        self.drop_overflow_update = drop_overflow_update

        assert isinstance(clip_grad, bool), f'Invalid type of clip_grad, got {type(clip_grad)}, expected bool'
        assert clip_norm > 0. and isinstance(clip_norm, float), f'clip_norm must be float > 1.0, but got {clip_norm}'
        self.clip_grad = clip_grad
        self.clip_norm = clip_norm

        self.verbose = verbose
        if self.ema:
            self.weights_all = ms.ParameterTuple(list(network.get_parameters()))
            self.ema_weight = self.weights_all.clone("ema", init="same")

        self.is_cpu_device = context.get_context("device_target") == 'CPU' # to support CPU in CI

        self.map = ops.Map()
        self.partial= ops.Partial()

    def ema_update(self):
        """Update EMA parameters."""
        self.updates += 1
        d = self.ema_decay * (1 - F.exp(-self.updates / 2000))
        # update trainable parameters
        success = self.hyper_map(F.partial(_ema_op, d), self.ema_weight, self.weights_all)
        self.updates = F.depend(self.updates, success)
        return self.updates

    def construct(self, *inputs):
        # compute loss
        weights = self.weights
        loss = self.network(*inputs) # mini-batch loss
        scaling_sens = self.scale_sense

        # check loss overflow
        if not self.is_cpu_device:
            status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        else:
            status = None

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss)) # loss scale value

        # compute gradients (of the up-scaled loss w.r.t. the model weights)
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)

        # down-scale gradidents with loss_scale value. TODO: divide scaling_sense by accumulation steps for grad accumulate
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)

        # gradient reduction on distributed GPUs/NPUs
        grads = self.grad_reducer(grads)

        # check gradient overflow
        if not self.is_cpu_device:
            cond = self.get_overflow_status(status, grads)
            overflow = self.process_loss_scale(cond)
        else:
            overflow = ms.Tensor(False)
            cond = ms.Tensor(False)

        # accumulate gradients and update model weights if no overflow or allow to update even when overflow
        if (not self.drop_overflow_update) or (not overflow):
            # clip grad
            if self.clip_grad:
                grads = ops.clip_by_global_norm(grads, self.clip_norm)

            # optimize
            loss = F.depend(loss, self.optimizer(grads))

            # EMA of model weights
            if self.ema:
                self.ema_update()
        else:
            #print("WARNING: Gradient overflow! update skipped.")
            pass

        return loss, cond, scaling_sens
