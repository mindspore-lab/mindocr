""" auto mixed precision related functions """

# from mindspore.amp import LossScaler  # this line of code leads to “get rank id error” in modelarts
from mindspore.common.api import ms_class


@ms_class
class LossScaler:
    r"""
    Loss scaler abstract class when using mixed precision.

    Derived class needs to implement all of its methods. During training, `scale` and `unscale` is used
    to scale and unscale the loss value and gradients to avoid overflow, `adjust` is used to update the
    loss scale value.

    Note:
        This is an experimental interface that is subject to change or deletion.
    """

    def scale(self, inputs):
        """
        Scaling inputs by `scale_value`.

        Args:
            inputs (Union(Tensor, tuple(Tensor))): the input loss value or gradients.
        """
        raise NotImplementedError

    def unscale(self, inputs):
        """
        Unscaling inputs by `scale_value`.

        Args:
            inputs (Union(Tensor, tuple(Tensor))): the input loss value or gradients.
        """
        raise NotImplementedError

    def adjust(self, grads_finite):
        """
        Adjust the `scale_value` dependent on whether grads are finite.

        Args:
            grads_finite (Tensor): a scalar bool Tensor indicating whether the grads are finite.
        """
        raise NotImplementedError


class NoLossScaler(LossScaler):
    """
    No LossScaler
    """

    def scale(self, inputs):
        return inputs

    def unscale(self, inputs):
        return inputs

    def adjust(self, grads_finite):
        return True
