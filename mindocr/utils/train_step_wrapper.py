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
        clip_grad=False, #TODO: adamw/lion opt also support clip grad. merge?
        clip_norm=1.0,
        gradient_accumulation_steps=1,
        verbose=False,
    ):
        super().__init__(network, optimizer, scale_sense)
        self.ema = ema
        self.ema_decay = ema_decay
        self.updates = Parameter(Tensor(updates, ms.float32), requires_grad=False)
        self.drop_overflow_update = drop_overflow_update

        assert isinstance(clip_grad, bool), f'Invalid type of clip_grad, got {type(clip_grad)}'
        assert clip_norm > 0. and isinstance(clip_norm, float), f'clip_norm must be float > 1.0, but got {clip_norm}'
        self.clip_grad = clip_grad
        self.clip_norm = clip_norm

        # Gradient accumulation
        assert gradient_accumulation_steps >= 1 and isinstance(gradient_accumulation_steps, int), f'gradient_accumulation_steps must be int >= 1, but got {gradient_accumulation_steps}'
        self.grad_accu_steps = gradient_accumulation_steps
        if self.grad_accu_steps > 1:
            # additionally caches network trainable parameters. overhead caused.
            # TODO: try to store it in CPU memory instead of GPU/NPU memory.
            self.accumulated_grads = optimizer.parameters.clone(prefix='grad_accumulated_', init='zeros')
            self.zeros = optimizer.parameters.clone(prefix='zeros_', init='zeros')
            for p in self.accumulated_grads:
                p.requires_grad = False
            for z in self.zeros:
                z.requires_grad = False
            self.cur_accu_step = Parameter(Tensor(0, ms.int32), 'grad_accumulate_step_', requires_grad=False)
            self.zero = Tensor(0, ms.int32)
        else:
            self.cur_accu_step  = 0 # it will allow to update model every step

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

        # up-scale loss with loss_scale value and gradient accumulation steps
        # NOTE: we choose to take mean over gradient accumulation steps here for the consistency with gradient accumulation implementation in pytorch.
        scaled_loss = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss)) / F.cast(self.grad_accu_steps, F.dtype(loss))

        # compute gradients
        grads = self.grad(self.network, weights)(*inputs, scaled_loss)

        # down-scale gradidents with loss_scale value only. (as a result, it is the same as dividing accumulated gradients with accumulation steps)
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

        #print(0, grads[0][0][0])

        # accumulate gradients and update model weights if no overflow or allow to update even when overflow
        if (not self.drop_overflow_update) or (not overflow):
            # gradient accumulation
            if self.grad_accu_steps > 1:
                success = F.depend(loss, self.map(self.partial(ops.assign_add), self.accumulated_grads, grads)) # self.accumulated_grads += grads
                success = F.depend(success, ops.assign_add(self.cur_accu_step, Tensor(1, ms.int32)))            # self.cur_accu_step += 1
                accu_grads = self.accumulated_grads
            else:
                success = loss
                accu_grads = grads

            # optimize
            # TODO: consider the last accumluation round, which is now skipped
            if self.cur_accu_step % self.grad_accu_steps == 0:
                #print(1, accu_grads[0][0][0])
                # clip grad
                if self.clip_grad:
                    clipped_grads = ops.clip_by_global_norm(accu_grads, self.clip_norm)
                else:
                    clipped_grads = accu_grads
                #print(2, clipped_grads[0][0][0])

                # NOTE: no need to divde accumulated grads with accumulation steps since we've divided loss with the steps.
                success = F.depend(success, self.optimizer(clipped_grads))

                # EMA of model weights
                #if self.ema:
                #    self.ema_update()

                # clear grad accumulation states
                if self.grad_accu_steps > 1:
                    success = F.depend(success, self.map(self.partial(ops.assign), self.accumulated_grads, self.zeros)) # self.accumulated_grads = 0
                    success = F.depend(success, ops.assign(self.cur_accu_step, self.zero))      # self.cur_accu_step = 0

        else:
            print("WARNING: Gradient overflow! update skipped.")
            pass

        return loss, cond, scaling_sens


    def _get_gradient_accumulation_fn(self):
        # code adopted from mindyolo
        hyper_map = ops.HyperMap()

        def accu_fn(g1, g2):
            g1 = g1 + g2
            return g1

        def gradient_accumulation_fn(accumulated_grads, grads):
            success = hyper_map(accu_fn, accumulated_grads, grads)
            return success

        return gradient_accumulation_fn


