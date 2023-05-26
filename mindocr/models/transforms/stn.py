from typing import Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor
from mindspore.ops.primitive import constexpr
from packaging import version

from .tps_spatial_transformer import TPSSpatialTransformer


@constexpr
def _if_version_is_large_than_two_point_zero():
    if version.parse(ms.__version__) >= version.parse("2.0.0rc"):
        return True
    return False


def conv3x3_block(
    in_channels: int, out_channels: int, stride: int = 1
) -> nn.SequentialCell:
    conv_layer = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        pad_mode="pad",
        padding=1,
        has_bias=False,
    )
    block = nn.SequentialCell(conv_layer, nn.BatchNorm2d(out_channels), nn.ReLU())
    return block


class STN(nn.Cell):
    def __init__(
        self, in_channels: int, num_ctrlpoints: int, activation: str = "none"
    ) -> None:
        super(STN, self).__init__()
        self.in_channels = in_channels
        self.num_ctrlpoints = num_ctrlpoints
        self.activation = activation
        self.stn_convnet = nn.SequentialCell(
            conv3x3_block(in_channels, 32),  # 32x64
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_block(32, 64),  # 16x32
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_block(64, 128),  # 8*16
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_block(128, 256),  # 4*8
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_block(256, 256),  # 2*4,
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_block(256, 256),
        )  # 1*2
        self.stn_fc1 = nn.SequentialCell(
            nn.Dense(2 * 256, 512), nn.BatchNorm1d(512), nn.ReLU()
        )
        fc2_bias = self.init_stn()
        self.stn_fc2 = nn.Dense(
            512, num_ctrlpoints * 2, weight_init="zeros", bias_init=fc2_bias
        )

    def init_stn(self) -> Tensor:
        margin = 0.01
        sampling_num_per_side = int(self.num_ctrlpoints / 2)
        ctrl_pts_x = np.linspace(margin, 1.0 - margin, sampling_num_per_side)
        ctrl_pts_y_top = np.ones(sampling_num_per_side) * margin
        ctrl_pts_y_bottom = np.ones(sampling_num_per_side) * (1 - margin)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        ctrl_points = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0).astype(
            np.float32
        )
        if self.activation == "none":
            pass
        elif self.activation == "sigmoid":
            ctrl_points = -np.log(1.0 / ctrl_points - 1.0)
        ctrl_points = Tensor(ctrl_points)
        fc2_bias = ops.reshape(ctrl_points, (-1,))
        return fc2_bias

    def construct(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.stn_convnet(x)
        batch_size = x.shape[0]
        x = ops.reshape(x, (batch_size, -1))
        img_feat = self.stn_fc1(x)
        x = self.stn_fc2(0.1 * img_feat)
        if self.activation == "sigmoid":
            x = ops.sigmoid(x)
        x = ops.reshape(x, (-1, self.num_ctrlpoints, 2))
        return img_feat, x


class STN_ON(nn.Cell):
    def __init__(
        self,
        in_channels: int = 3,
        tps_inputsize: Tuple[int, int] = [32, 64],
        tps_outputsize: Tuple[int, int] = [32, 100],
        num_control_points: int = 20,
        tps_margins: Tuple[float, float] = [0.05, 0.05],
        stn_activation: str = "none",
    ) -> None:
        super(STN_ON, self).__init__()
        self.tps = TPSSpatialTransformer(
            output_image_size=tuple(tps_outputsize),
            num_control_points=num_control_points,
            margins=tuple(tps_margins),
        )
        self.stn_head = STN(
            in_channels=in_channels,
            num_ctrlpoints=num_control_points,
            activation=stn_activation,
        )
        self.tps_inputsize = tuple(tps_inputsize)
        self.out_channels = in_channels

    def construct(self, image: Tensor) -> Tensor:
        if _if_version_is_large_than_two_point_zero():
            stn_input = ops.interpolate(image, size=self.tps_inputsize, mode="bilinear")
        else:
            stn_input = ops.interpolate(
                image, sizes=self.tps_inputsize, mode="bilinear"
            )
        _, ctrl_points = self.stn_head(stn_input)
        x, _ = self.tps(image, ctrl_points)
        return x
