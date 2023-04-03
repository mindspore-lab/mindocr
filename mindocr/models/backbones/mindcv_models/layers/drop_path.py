"""DropPath
Mindspore implementations of DropPath (Stochastic Depth) regularization layers.
Papers:
Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)
"""
from mindspore import Tensor, nn, ops
from mindspore.numpy import ones


class DropPath(nn.Cell):
    """DropPath (Stochastic Depth) regularization layers"""

    def __init__(
        self,
        drop_prob: float = 0.0,
        scale_by_keep: bool = True,
    ) -> None:
        super().__init__()
        self.keep_prob = 1.0 - drop_prob
        self.scale_by_keep = scale_by_keep
        self.dropout = nn.Dropout(self.keep_prob)

    def construct(self, x: Tensor) -> Tensor:
        if self.keep_prob == 1.0 or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.dropout(ones(shape))
        if not self.scale_by_keep:
            random_tensor = ops.mul(random_tensor, self.keep_prob)
        return x * random_tensor
