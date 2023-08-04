import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from ..backbones.mindcv_models.layers.pooling import GlobalAvgPooling


class ClsHead(nn.Cell):
    """
    Dense Head for classification
    """

    def __init__(self, in_channels: int, num_classes: int, **kwargs) -> None:
        super(ClsHead, self).__init__()
        self.pool = GlobalAvgPooling()
        self.fc = nn.Dense(in_channels, num_classes)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x: Tensor) -> Tensor:
        N, W, C = x.shape
        x = ops.transpose(x, (0, 2, 1))
        x = ops.reshape(x, (N, C, -1, W))
        x = self.pool(x)
        x = self.fc(x)
        if not self.training:
            x = self.softmax(x)
        return x


class MobileNetV3Head(nn.Cell):
    """
    Text direction classification head for MobileNetV3.
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
