import mindspore.nn as nn


class CTCHead(nn.Cell):
    def __init__(self, in_channels,
                 out_channels,
                 #  return_feats=False,
                 **kwargs):
        super(CTCHead, self).__init__()
        self.fc = nn.Dense(in_channels, out_channels)

    def construct(self, x):
        result = self.fc(x)
        return result
