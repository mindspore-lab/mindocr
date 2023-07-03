from typing import Tuple, Union

import mindspore as ms
import mindspore.nn as nn


class DBHead(nn.Cell):
    """
    Differentiable Binarization model's head described in `DBNet <https://arxiv.org/abs/1911.08947>`__ paper.
    Predicts probability (or segmentation) map along with threshold and approximate binary maps (last two are optional).

    Args:
        in_channels: Number of input channels.
        k: Amplification factor for the approximate binarization step. Default: 50
        adaptive: Whether to produce threshold and approximate binary maps during training (recommended).
                  Inactive during inference (to save computational time). Default: True.
        bias: Use bias in Conv2d operations. Default: False.
        weight_init: Weights initialization method. Default: 'HeUniform'.
    """
    def __init__(self, in_channels: int, k: int = 50, adaptive: bool = True, bias: bool = False,
                 weight_init: str = 'HeUniform'):
        super().__init__()
        self.adaptive = adaptive

        self.segm = self._init_heatmap(in_channels, in_channels // 4, weight_init, bias)
        if self.adaptive:
            self.thresh = self._init_heatmap(in_channels, in_channels // 4, weight_init, bias)
            self.k = k
            self.diff_bin = nn.Sigmoid()

    @staticmethod
    def _init_heatmap(in_channels: int, inter_channels: int, weight_init: str, bias: bool) -> nn.SequentialCell:
        return nn.SequentialCell([  # `pred` block from the original work
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=bias,
                      weight_init=weight_init),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2dTranspose(inter_channels, inter_channels, kernel_size=2, stride=2, pad_mode='valid', has_bias=True,
                               weight_init=weight_init),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2dTranspose(inter_channels, 1, kernel_size=2, stride=2, pad_mode='valid', has_bias=True,
                               weight_init=weight_init),
            nn.Sigmoid()
        ])

    def construct(self, features: ms.Tensor) -> Union[ms.Tensor, Tuple[ms.Tensor, ...]]:
        binary = self.segm(features)

        if self.adaptive and self.training:     # use the binary map only to derive polygons during inference
            thresh = self.thresh(features)
            thresh_binary = self.diff_bin(self.k * binary - thresh)  # Differentiable Binarization
            return binary, thresh, thresh_binary

        return binary
