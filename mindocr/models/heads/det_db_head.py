import mindspore.nn as nn


class DBHead(nn.Cell):
    def __init__(self, in_channels: int, k=50, adaptive=False, bias=False, weight_init='HeUniform'):
        super().__init__()
        self.adaptive = adaptive

        self.segm = self._init_heatmap(in_channels, in_channels // 4, weight_init, bias)
        if adaptive:
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

    def construct(self, features):
        pred = {'binary': self.segm(features)}

        if self.adaptive:
            pred['thresh'] = self.thresh(features)
            pred['thresh_binary'] = self.diff_bin(
                self.k * (pred['binary'] - pred['thresh']))  # Differentiable Binarization

        return pred
