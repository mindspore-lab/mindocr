import mindspore.nn as nn
from mindspore import Tensor

from ..backbones.mindcv_models.layers.pooling import GlobalAvgPooling


class ClsHead(nn.Cell):
    """
    Text direction classification head.
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int):
        super().__init__()
        self.pool = GlobalAvgPooling()
        self.classifier = nn.SequentialCell([
            nn.Dense(in_channels, hidden_channels),
            nn.HSwish(),
            nn.Dropout(keep_prob=0.8),
            nn.Dense(hidden_channels, num_classes),
        ])
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = x.astype('float32')
        x = self.classifier(x)
        x = self.softmax(x)
        return x
