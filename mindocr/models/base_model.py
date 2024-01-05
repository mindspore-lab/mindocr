from addict import Dict

from mindspore import nn

from .backbones import build_backbone
from .heads import build_head
from .necks import build_neck
from .transforms import build_trans

__all__ = ["BaseModel"]


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
        if config.type == "kie":
            self.is_kie = True
        else:
            self.is_kie = False

        if config.transform:
            transform_name = config.transform.pop("name")
            self.transform = build_trans(transform_name, **config.transform)
        else:
            self.transform = None

        backbone_name = config.backbone.pop("name")
        self.backbone = build_backbone(backbone_name, **config.backbone)

        assert hasattr(self.backbone, "out_channels"), (
            f"Backbones are required to provide out_channels attribute, "
            f"but not found in {backbone_name}"
        )
        if "neck" not in config or config.neck is None:
            neck_name = "Select"
        else:
            neck_name = config.neck.pop("name")
        if self.is_kie:
            neck_name = "Identity"
        self.neck = build_neck(
            neck_name, in_channels=self.backbone.out_channels, **config.neck
        )

        assert hasattr(self.neck, "out_channels"), (
            f"Necks are required to provide out_channels attribute, "
            f"but not found in {neck_name}"
        )

        self.head_name = config.head.pop("name")
        self.head = build_head(self.head_name, in_channels=self.neck.out_channels, **config.head)

        self.model_name = f"{backbone_name}_{neck_name}_{self.head_name}"

    def ser(self, *inputs):
        image = inputs[4]

        x = self.backbone(
            input_ids=inputs[0],
            bbox=inputs[1],
            attention_mask=inputs[2],
            token_type_ids=inputs[3],
            pixel_values=image,
        )
        x = self.head(x, inputs[0])

        return x

    def re(self, *inputs):
        if self.backbone.use_visual_backbone is True:
            image = inputs[4]
        else:
            image = None

        x = self.backbone(
            input_ids=inputs[0],
            bbox=inputs[1],
            attention_mask=inputs[2],
            token_type_ids=inputs[3],
            image=image,
        )
        x = self.head(x, inputs[0], inputs[5], inputs[6], inputs[7], inputs[8])

        return x

    def construct(self, *args):
        if self.is_kie is True:
            if self.head_name == "TokenClassificationHead":
                return self.ser(*args)
            elif self.head_name == "RelationExtractionHead":
                return self.re(*args)

        x = args[0]
        if self.transform is not None:
            x = self.transform(x)

        # TODO: return bout, hout for debugging, using a dict.
        bout = self.backbone(x)

        nout = self.neck(bout)

        if len(args) > 1:
            hout = self.head(nout, args[1:])
        else:
            hout = self.head(nout)

        # resize back for postprocess
        # y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)

        return hout


if __name__ == "__main__":
    model_config = {
        "backbone": {"name": "det_resnet50", "pretrained": False},
        "neck": {
            "name": "FPN",
            "out_channels": 256,
        },
        "head": {"name": "ConvHead", "out_channels": 2, "k": 50},
    }
    model_config.pop("neck")
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
        print(time.time() - start)
        print(y.shape)

    predict(model, x)
