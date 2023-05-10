from mindspore import nn, ops


class AdaptiveScaleFusion(nn.Cell):
    """
    Adaptive Scale Fusion module from the `DBNet++ <https://arxiv.org/abs/2202.10304>`__ paper.
    Args:
        channels: number of input to and output channels from ASF
        channel_attention: use channel attention
    """
    def __init__(self, channels, channel_attention=True, weight_init='HeUniform'):
        super().__init__()
        out_channels = channels // 4
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=True,
                              weight_init=weight_init)

        self.chan_att = nn.SequentialCell([
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, pad_mode='valid', weight_init=weight_init),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, pad_mode='valid', weight_init=weight_init),
            nn.Sigmoid()
        ]) if channel_attention else None

        self.spat_att = nn.SequentialCell([
            nn.Conv2d(1, 1, kernel_size=3, padding=1, pad_mode='pad', weight_init=weight_init),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=1, pad_mode='valid', weight_init=weight_init),
            nn.Sigmoid()
        ])

        self.scale_att = nn.SequentialCell([
            nn.Conv2d(out_channels, 4, kernel_size=1, pad_mode='valid', weight_init=weight_init),
            nn.Sigmoid()
        ])

    def construct(self, x):
        reduced = self.conv(ops.concat(x, axis=1))

        if self.chan_att is not None:
            ada_pool = ops.mean(reduced, axis=(-2, -1), keep_dims=True)  # equivalent to nn.AdaptiveAvgPool2d(1)
            reduced = self.chan_att(ada_pool) + reduced

        spatial = ops.mean(reduced, axis=1, keep_dims=True)
        spat_att = self.spat_att(spatial) + reduced

        scale_att = self.scale_att(spat_att)
        return ops.concat([scale_att[:, i:i + 1] * x[i] for i in range(len(x))], axis=1)
