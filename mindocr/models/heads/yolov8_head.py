import math
import os

import numpy as np

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Parameter, Tensor, nn, ops

from mindocr.models.utils.yolov8_cells import DFL, ConvNormAct, Identity, meshgrid

__all__ = ["YOLOv8Head", "yolov8_head"]


class YOLOv8Head(nn.Cell):
    # YOLOv8 Detect head for detection models
    def __init__(self, nc=80, reg_max=16, stride=(), in_channels=(), sync_bn=False):  # detection layer
        super().__init__()

        assert isinstance(stride, (tuple, list)) and len(stride) > 0
        assert isinstance(in_channels, (tuple, list)) and len(in_channels) > 0

        self.nc = nc  # number of classes
        self.nl = len(in_channels)  # number of detection layers
        self.reg_max = reg_max  # DFL channels (in_channels[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = Parameter(Tensor(stride, ms.int32), requires_grad=False)

        c2, c3 = max((16, in_channels[0] // 4, self.reg_max * 4)), max(in_channels[0], self.nc)  # channels
        self.cv2 = nn.CellList(
            [
                nn.SequentialCell(
                    [
                        ConvNormAct(x, c2, 3, sync_bn=sync_bn),
                        ConvNormAct(c2, c2, 3, sync_bn=sync_bn),
                        nn.Conv2d(c2, 4 * self.reg_max, 1, has_bias=True),
                    ]
                )
                for x in in_channels
            ]
        )
        self.cv3 = nn.CellList(
            [
                nn.SequentialCell(
                    [
                        ConvNormAct(x, c3, 3, sync_bn=sync_bn),
                        ConvNormAct(c3, c3, 3, sync_bn=sync_bn),
                        nn.Conv2d(c3, self.nc, 1, has_bias=True),
                    ]
                )
                for x in in_channels
            ]
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else Identity()

        # reset parameter for Detect Head
        self.initialize_biases()
        self.dfl.initialize_conv_weight()

    def construct(self, x):
        shape = x[0].shape  # BCHW
        out = ()
        for i in range(self.nl):
            out += (ops.concat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1),)

        p = None
        if not self.training:
            _anchors, _strides = self.make_anchors(out, self.stride, 0.5)
            _anchors, _strides = _anchors.swapaxes(0, 1), _strides.swapaxes(0, 1)
            _x = ()
            for i in range(len(out)):
                _x += (out[i].view(shape[0], self.no, -1),)
            _x = ops.concat(_x, 2)
            box, cls = _x[:, : self.reg_max * 4, :], _x[:, self.reg_max * 4: self.reg_max * 4 + self.nc, :]
            dbox = self.dist2bbox(self.dfl(box), ops.expand_dims(_anchors, 0), xywh=True, axis=1) * _strides
            p = ops.concat((dbox, ops.Sigmoid()(cls)), 1)
            p = ops.transpose(p, (0, 2, 1))  # (bs, no-84, nbox) -> (bs, nbox, no-84)

        return out if self.training else p

    @staticmethod
    def make_anchors(feats, strides, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points, stride_tensor = (), ()
        dtype = feats[0].dtype
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = mnp.arange(w, dtype=dtype) + grid_cell_offset  # shift x
            sy = mnp.arange(h, dtype=dtype) + grid_cell_offset  # shift y
            sy, sx = meshgrid((sy, sx), indexing="ij")
            anchor_points += (ops.stack((sx, sy), -1).view(-1, 2),)
            stride_tensor += (ops.ones((h * w, 1), dtype) * stride,)
        return ops.concat(anchor_points), ops.concat(stride_tensor)

    @staticmethod
    def dist2bbox(distance, anchor_points, xywh=True, axis=-1):
        """Transform distance(ltrb) to box(xywh or xyxy)."""
        lt, rb = ops.split(distance, split_size_or_sections=2, axis=axis)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return ops.concat((c_xy, wh), axis)  # xywh bbox
        return ops.concat((x1y1, x2y2), axis)  # xyxy bbox

    def initialize_biases(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            s = s.asnumpy()
            a[-1].bias = ops.assign(a[-1].bias, Tensor(np.ones(a[-1].bias.shape), ms.float32))
            b_np = b[-1].bias.data.asnumpy()
            b_np[: m.nc] = math.log(5 / m.nc / (640 / int(s)) ** 2)
            b[-1].bias = ops.assign(b[-1].bias, Tensor(b_np, ms.float32))


def yolov8_head(nc=5, reg_max=16, stride=None, in_channels=None) -> YOLOv8Head:
    if stride is None:
        stride = [8, 16, 32, 64]
    if in_channels is None:
        in_channels = [64, 128, 192, 256]
    model = YOLOv8Head(nc=nc, reg_max=reg_max, stride=stride, in_channels=in_channels)
    return model


def test_yolov8_head():
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device("Ascend", os.environ.get("DEVICE_ID", 0))
    ms.set_seed(0)

    network = yolov8_head()
    print(network)

    inputs = [
        Tensor(np.random.randn(1, 64, 100, 100), ms.float32),
        Tensor(np.random.randn(1, 128, 50, 50), ms.float32),
        Tensor(np.random.randn(1, 192, 25, 25), ms.float32),
        Tensor(np.random.randn(1, 256, 13, 13), ms.float32),
    ]
    out = network(inputs)
    print(out)

    for o in out:
        if isinstance(o, Tensor):
            print(o.shape)
        elif isinstance(o, (tuple, list)):
            for oo in o:
                print(oo.shape)
        else:
            print(o)


if __name__ == '__main__':
    test_yolov8_head()
