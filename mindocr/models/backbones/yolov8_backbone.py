import os

import numpy as np

import mindspore as ms
from mindspore import Tensor, nn

from mindocr.models.backbones import register_backbone
from mindocr.models.utils.yolov8_cells import Model, YOLOv8BaseConfig, initialize_default

__all__ = ["YOLOv8Backbone", "yolov8_backbone"]


class YOLOv8BackboneConfig(YOLOv8BaseConfig):
    def __init__(self,
                 backbone=None,
                 nc=5,
                 reg_max=16,
                 stride=None,
                 depth_multiple=1.0,
                 width_multiple=1.0,
                 max_channels=1024,
                 sync_bn=False,
                 ):
        super(YOLOv8BackboneConfig, self).__init__(nc=nc,
                                                   reg_max=reg_max,
                                                   stride=stride,
                                                   depth_multiple=depth_multiple,
                                                   width_multiple=width_multiple,
                                                   max_channels=max_channels,
                                                   sync_bn=sync_bn)
        if backbone is None:
            backbone = [
                [-1, 1, 'ConvNormAct', [64, 3, 2]],  # 0-P1/2
                [-1, 1, 'ConvNormAct', [128, 3, 2]],  # 1-P2/4
                [-1, 3, 'C2f', [128, True]],
                [-1, 1, 'ConvNormAct', [256, 3, 2]],  # 3-P3/8
                [-1, 6, 'C2f', [256, True]],
                [-1, 1, 'ConvNormAct', [512, 3, 2]],  # 5-P4/16
                [-1, 6, 'C2f', [512, True]],
                [-1, 1, 'ConvNormAct', [768, 3, 2]],  # 7-P5/32
                [-1, 3, 'C2f', [768, True]],
                [-1, 1, 'ConvNormAct', [1024, 3, 2]],  # 9-P6/64
                [-1, 3, 'C2f', [1024, True]],
                [-1, 1, 'SPPF', [1024, 5]],
                [-1, 1, 'Upsample', ['None', 1.95, 'nearest']],  # 1.95 suitable for images with img_size of 640 or 800.
                [[-1, 8], 1, 'Concat', [1]],
                [-1, 3, 'C2f', [768, False]],
                [-1, 1, 'Upsample', ['None', 2, 'nearest']],
                [[-1, 6], 1, 'Concat', [1]],
                [-1, 3, 'C2f', [512, False]],
                [-1, 1, 'Upsample', ['None', 2, 'nearest']],
                [[-1, 4], 1, 'Concat', [1]],
                [-1, 3, 'C2f', [256, False]],  # 20
                [-1, 1, 'ConvNormAct', [256, 3, 2]],
                [[-1, 17], 1, 'Concat', [1]],
                [-1, 3, 'C2f', [512, False]],  # 23
                [-1, 1, 'ConvNormAct', [512, 3, 2]],
                [[-1, 14], 1, 'Concat', [1]],
                [-1, 3, 'C2f', [768, False]],  # 26
                [-1, 1, 'ConvNormAct', [768, 3, 2]],
                [[-1, 11], 1, 'Concat', [1]],
                [-1, 3, 'C2f', [1024, False]],  # 29
            ]
        self.backbone = backbone


class YOLOv8Backbone(nn.Cell):
    def __init__(self, cfg=None, in_channels=3, out_channels=None):
        super(YOLOv8Backbone, self).__init__()
        if cfg is None:
            cfg = YOLOv8BackboneConfig()
        self.cfg = cfg
        self.stride = Tensor(np.array(cfg.stride), ms.int32)
        self.stride_max = int(max(self.cfg.stride))
        ch, nc = in_channels, cfg.nc

        self.nc = nc  # override yaml value
        if out_channels is None:
            out_channels = [64, 128, 192, 256]
        self.out_channels = out_channels
        self.model = Model(model_cfg=cfg, in_channels=ch)
        self.names = [str(i) for i in range(nc)]  # default names

        self.reset_parameter()

    def construct(self, x):
        return self.model(x)

    def reset_parameter(self):
        # init default
        initialize_default(self)


@register_backbone
def yolov8_backbone(
        backbone=None,
        nc=5,
        reg_max=16,
        stride=None,
        depth_multiple=1.0,
        width_multiple=1.0,
        max_channels=1024,
        sync_bn=False,
        out_channels=None,
        pretrained=None,
) -> YOLOv8Backbone:
    cfg = YOLOv8BackboneConfig(backbone=backbone,
                               nc=nc,
                               reg_max=reg_max,
                               stride=stride,
                               depth_multiple=depth_multiple,
                               width_multiple=width_multiple,
                               max_channels=max_channels,
                               sync_bn=sync_bn)
    if out_channels is None:
        out_channels = [64, 128, 192, 256]
    model = YOLOv8Backbone(cfg=cfg, in_channels=3, out_channels=out_channels)
    return model


def test_yolo_backbone():
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device("Ascend", os.environ.get("DEVICE_ID", 0))
    ms.set_seed(0)

    network = YOLOv8Backbone()
    print(network)

    x = Tensor(np.random.randn(1, 3, 800, 800), ms.float32)
    out = network(x)
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
    test_yolo_backbone()
