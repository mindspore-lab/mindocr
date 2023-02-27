import mindspore as ms
import mindspore.nn as nn
from mindspore.common.initializer import HeNormal
from mindspore.common import initializer as init

from .asf import ScaleFeatureSelection

from mindspore import ops

class FPN(nn.Cell):
    def __init__(self, in_channels, out_channels=256, **kwargs):
        super().__init__()
        self.out_channels = out_channels

    def construct(self, x):
        x1, x2, x3, x4 = x
        return x1

class DBFPN(nn.Cell):
    def __init__(self, in_channels, out_channels=256, 
                 bias=False, use_asf=False, attention_type='scale_channel_spatial'):
        '''
        in_channels: resnet18=[64, 128, 256, 512]
                    resnet50=[2048,1024,512,256]
        out_channels: Inner channels in Conv2d

        bias: Whether conv layers have bias or not.
        use_asf: use ASF moduel for multi-scale feature aggregation
        '''

        super().__init__()

        self.out_channels = out_channels

        self.in5 = nn.Conv2d(in_channels[-1], out_channels, 1, has_bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], out_channels, 1, has_bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], out_channels, 1, has_bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], out_channels, 1, has_bias=bias)

        self.out5 = nn.Conv2d(out_channels, out_channels // 4, 3, pad_mode="pad", padding=1, has_bias=bias)
        self.out4 = nn.Conv2d(out_channels, out_channels // 4, 3, pad_mode="pad", padding=1, has_bias=bias)
        self.out3 = nn.Conv2d(out_channels, out_channels // 4, 3, pad_mode="pad", padding=1, has_bias=bias)
        self.out2 = nn.Conv2d(out_channels, out_channels // 4, 3, pad_mode="pad", padding=1, has_bias=bias)
        
        self.use_asf = use_asf
        if use_asf:
            self.concat_attention = ScaleFeatureSelection(inner_channels, inner_channels // 4,
                                                          attention_type=attention_type)
            self.concat_attention.weights_init(self.concat_attention)

        self.weights_init(self.in5)
        self.weights_init(self.in4)
        self.weights_init(self.in3)
        self.weights_init(self.in2)

        self.weights_init(self.out5)
        self.weights_init(self.out4)
        self.weights_init(self.out3)
        self.weights_init(self.out2)

    def weights_init(self, c):
        for m in c.cells():
            if isinstance(m, nn.Conv2dTranspose):
                m.weight = init.initializer(HeNormal(), m.weight.shape)
                m.bias = init.initializer('zeros', m.bias.shape)

            elif isinstance(m, nn.Conv2d):
                m.weight = init.initializer(HeNormal(), m.weight.shape)

            elif isinstance(m, nn.BatchNorm2d):
                m.gamma = init.initializer('ones', m.gamma.shape)
                m.beta = init.initializer(1e-4, m.beta.shape)

    def construct(self, features):

        # shapes for inference:
        # [1, 64, 184, 320]
        # [1, 128, 92, 160]
        # [1, 256, 46, 80]
        # [1, 512, 23, 40]

        c2, c3, c4, c5 = features

        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        # Carry out up sampling and prepare for connection
        up5 = ops.ResizeNearestNeighbor((in4.shape[2], in4.shape[3]))
        up4 = ops.ResizeNearestNeighbor((in3.shape[2], in3.shape[3]))
        up3 = ops.ResizeNearestNeighbor((in2.shape[2], in2.shape[3]))

        out4 = up5(in5) + in4  # 1/16
        out3 = up4(out4) + in3  # 1/8
        out2 = up3(out3) + in2  # 1/4

        upsample = ops.ResizeNearestNeighbor((c2.shape[2], c2.shape[3]))

        # The connected results are upsampled to make them the same shape, 1/4
        p5 = upsample(self.out5(in5))
        p4 = upsample(self.out4(out4))
        p3 = upsample(self.out3(out3))
        p2 = upsample(self.out2(out2))

        fuse = ops.Concat(1)((p5, p4, p3, p2))
        if self.use_asf:
            fuse = self.concat_attention(fuse, [p5, p4, p3, p2])

        return fuse
