from typing import Tuple, Union

import mindspore as ms
import mindspore.nn as nn


class DBHead(nn.Cell):
    def __init__(self, in_channels: int, k=50, adaptive=False, bias=False, weight_init='HeUniform'):
        super().__init__()
        self.adaptive = adaptive

        self.segm = self._init_heatmap(in_channels, in_channels // 4, weight_init, bias)
        if self.adaptive:
            self.thresh = self._init_heatmap(in_channels, in_channels // 4, weight_init, bias)
            self.k = k
            self.diff_bin = nn.Sigmoid()

    @staticmethod
    def _init_heatmap(in_channels, inter_channels, weight_init, bias):
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
        """
        Args:
            features (Tensor): encoded features
        Returns:
            Union(
            binary: predicted binary map
            thresh: predicted threshold map (only return if adaptive is True in training)
            thresh_binary: differentiable binary map (only if adaptive is True in training)
        """
        binary = self.segm(features)

        if self.adaptive and self.training:
            # only use binary map to derive polygons in inference
            thresh = self.thresh(features)
            thresh_binary = self.diff_bin(self.k * binary - thresh)  # Differentiable Binarization
            return binary, thresh, thresh_binary

        return binary
