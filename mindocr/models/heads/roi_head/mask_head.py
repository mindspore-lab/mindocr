from mindspore import nn
from mindspore.common.initializer import HeNormal, Normal


class MaskRCNNConvUpSampleHead(nn.SequentialCell):

    def __init__(self, in_channels, num_classes=5, conv_dims=[]):
        super().__init__()

        cur_channels = in_channels
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = nn.Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                padding=1,
                pad_mode='pad',
                weight_init=HeNormal(mode="fan_out", nonlinearity="relu"),
                has_bias=True,
                bias_init="zeros"
            )
            self.insert_child_to_cell("mask_fcn{}".format(k + 1), conv)
            cur_channels = conv_dim

        self.deconv = nn.Conv2dTranspose(
            in_channels=cur_channels,
            out_channels=conv_dims[-1],
            kernel_size=2,
            stride=2,
            pad_mode="valid",
            weight_init=HeNormal(mode="fan_out", nonlinearity="relu"),
            has_bias=True,
            bias_init="zeros"
        )
        self.insert_child_to_cell("deconv_relu", nn.ReLU())
        cur_channels = conv_dims[-1]

        self.predictor = nn.Conv2d(
            cur_channels,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            pad_mode='valid',
            weight_init=Normal(sigma=0.001),
            has_bias=True,
            bias_init="zeros"
        )

    def construct(self, x):
        for layer in self:
            x = layer(x)
        return x
