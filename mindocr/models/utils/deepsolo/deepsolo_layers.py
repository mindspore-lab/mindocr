import math

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import numpy as mnp
from mindspore import ops


class PositionEmbedding2D(nn.Cell):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = Tensor(temperature)
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def construct(self, mask):
        """construct"""
        not_mask = ~mask.astype("bool")
        y_embed = ops.cumsum(not_mask, 1, dtype=mstype.float32)
        x_embed = ops.cumsum(not_mask, 2, dtype=mstype.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = mnp.arange(self.num_pos_feats, dtype=mstype.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = ops.Stack(axis=4)((ops.Sin()(pos_x[:, :, :, 0::2]), ops.Cos()(pos_x[:, :, :, 1::2])))
        pos_x = pos_x.view(*pos_x.shape[:3], -1)
        pos_y = ops.Stack(axis=4)((ops.Sin()(pos_y[:, :, :, 0::2]), ops.Cos()(pos_y[:, :, :, 1::2])))
        pos_y = pos_y.view(*pos_y.shape[:3], -1)
        pos = ops.Concat(axis=3)((pos_y, pos_x)).transpose(0, 3, 1, 2)
        return pos
        # return {"not_mask": not_mask, "x_embed": x_embed, "y_embed": y_embed, "dim_t": dim_t, "pos_x": pos_x, "pos_y": pos_y, "pos": pos}

