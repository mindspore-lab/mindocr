from typing import Tuple
from mindspore import Tensor, nn, ops

from .asf import AdaptiveScaleFusion


def _resize_nn(x: Tensor, scale: int = 0, shape: Tuple[int] = None):
    if scale == 1 or shape == x.shape[2:]:
        return x

    if scale:
        shape = (x.shape[2] * scale, x.shape[3] * scale)
    return ops.ResizeNearestNeighbor(shape)(x)


class FPN(nn.Cell):
    def __init__(self, in_channels, out_channels=256, **kwargs):
        super().__init__()
        self.out_channels = out_channels

    def construct(self, x):
        x1, x2, x3, x4 = x
        return x1


class DBFPN(nn.Cell):
    def __init__(self, in_channels, out_channels=256, weight_init='HeUniform',
                 bias=False, use_asf=False, channel_attention=True):
        """
        in_channels: resnet18=[64, 128, 256, 512]
                    resnet50=[2048,1024,512,256]
        out_channels: Inner channels in Conv2d

        bias: Whether conv layers have bias or not.
        use_asf: use ASF module for multi-scale feature aggregation (DBNet++ only)
        channel_attention: use channel attention in ASF module
        """
        super().__init__()
        self.out_channels = out_channels

        self.unify_channels = nn.CellList(
            [nn.Conv2d(ch, out_channels, 1, pad_mode='valid', has_bias=bias, weight_init=weight_init)
             for ch in in_channels]
        )

        self.out = nn.CellList(
            [nn.Conv2d(out_channels, out_channels // 4, 3, padding=1, pad_mode='pad', has_bias=bias,
                       weight_init=weight_init) for _ in range(len(in_channels))]
        )

        self.fuse = AdaptiveScaleFusion(out_channels, channel_attention, weight_init) if use_asf else ops.Concat(axis=1)

    def construct(self, features):
        for i, uc_op in enumerate(self.unify_channels):
            features[i] = uc_op(features[i])

        for i in range(2, -1, -1):
            features[i] += _resize_nn(features[i + 1], shape=features[i].shape[2:])

        for i, out in enumerate(self.out):
            features[i] = _resize_nn(out(features[i]), shape=features[0].shape[2:])

        return self.fuse(features[::-1])   # matching the reverse order of the original work
