from addict import Dict

from mindspore import nn

from .backbones import build_backbone
from .heads import build_head
from .necks import build_neck
from .transforms import build_trans

__all__ = ['BaseModel']


class BaseModel(nn.Cell):
    def __init__(self, config: dict):
        """
        Args:
            config (dict): model config

        Inputs:
            x (Tensor): The input tensor feeding into the backbone, neck and head sequentially.
            y (Tensor): The extra input tensor. If it is provided, it will feed into the head. Default: None
        """
        super(BaseModel, self).__init__()

        config = Dict(config)
        self.type = config.type
        if config.transform:
            transform_name = config.transform.pop('name')
            self.transform = build_trans(transform_name, **config.transform)
        else:
            self.transform = None

        backbone_name = config.backbone.pop('name')
        self.backbone = build_backbone(backbone_name, **config.backbone)

        if self.type == "kie":
            self.neck = None
            self.head = None
            self.model_name = backbone_name
        else:
            assert hasattr(self.backbone, 'out_channels'), f'Backbones are required ' \
                                                           f'to provide out_channels attribute, ' \
                                                           f'but not found in {backbone_name}'

            if 'neck' not in config or config.neck is None:
                neck_name = 'Select'
            else:
                neck_name = config.neck.pop('name')
            self.neck = build_neck(neck_name, in_channels=self.backbone.out_channels, **config.neck)

            assert hasattr(self.neck, 'out_channels'), f'Necks are required to provide out_channels attribute, ' \
                                                       f'but not found in {neck_name}'

            head_name = config.head.pop('name')
            self.head = build_head(head_name, in_channels=self.neck.out_channels, **config.head)

            self.model_name = f'{backbone_name}_{neck_name}_{head_name}'

    def construct(self, *args):
        if self.type == "kie":
            x = args
        else:
            x = args[0]

        if self.transform is not None:
            x = self.transform(x)

        # TODO: return bout, hout for debugging, using a dict.
        x = self.backbone(x)

        if self.neck is not None:
            x = self.neck(x)

        if self.head is not None:
            if len(args) > 1:
                x = self.head(x, args[1:])
            else:
                x = self.head(x)
        # resize back for postprocess
        # y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)

        return x


if __name__ == '__main__':
    model_config = {
            "backbone": {
                'name': 'det_resnet50',
                'pretrained': False
                },
            "neck": {
                "name": 'FPN',
                "out_channels": 256,
                },
            "head": {
                "name": 'ConvHead',
                "out_channels": 2,
                "k": 50
                }

            }
    model_config.pop('neck')
    model = BaseModel(model_config)

    import time

    import numpy as np

    import mindspore as ms

    bs = 8
    x = ms.Tensor(np.random.rand(bs, 3, 640, 640), dtype=ms.float32)
    ms.set_context(mode=ms.PYNATIVE_MODE)

    def predict(model, x):
        start = time.time()
        y = model(x)
        print(time.time()-start)
        print(y.shape)

    predict(model, x)
