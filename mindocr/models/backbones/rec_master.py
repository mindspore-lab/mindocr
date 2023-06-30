from typing import Any, Dict, List, Optional, Type

from mindspore import Tensor, nn

from ._registry import register_backbone, register_backbone_class
from .blocks import MultiAspectGCAttention

__all__ = ["rec_resnet_master_resnet31", "RecResNetMaster"]


class BasicBlockGCAtten(nn.Cell):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm: Optional[nn.Cell] = None,
        down_sample: Optional[nn.Cell] = None,
        use_gcb: bool = False,
        gcb_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            pad_mode="pad",
        )
        self.bn1 = norm(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            pad_mode="pad",
        )
        self.bn2 = norm(out_channels)
        self.down_sample = down_sample

        self.use_gcb = use_gcb
        if self.use_gcb:
            gcb_ratio = gcb_config["ratio"]
            gcb_headers = gcb_config["headers"]
            att_scale = gcb_config["att_scale"]
            fusion_type = gcb_config["fusion_type"]
            self.context_block = MultiAspectGCAttention(
                inplanes=out_channels,
                ratio=gcb_ratio,
                headers=gcb_headers,
                att_scale=att_scale,
                fusion_type=fusion_type,
            )

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.use_gcb:
            x = self.context_block(x)

        if self.down_sample is not None:
            identity = self.down_sample(identity)

        x += identity
        x = self.relu(x)
        return x


@register_backbone_class
class RecResNetMaster(nn.Cell):
    """MASTER Backbone, based on
    `"MASTER: Multi-Aspect Non-local Network for Scene Text Recognition"
    <https://arxiv.org/abs/2205.00159>`_.

    Args:
        block: Type of the block, support BasicBlockGCAtten only.
        layers: Numeber of the layers in teach block.
        in_channels: Number of the input channels. Default: 3.
        norm: Normalization method. If it is None, then BatchNorm2d will be used. Default: None.
        gcb_config: Configurations of the CGB block. If it is None, then no GCB block is applied. Default: None.
    """
    def __init__(
        self,
        block: Type[BasicBlockGCAtten],
        layers: List[int],
        in_channels: int = 3,
        norm: Optional[Type[nn.Cell]] = None,
        gcb_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d

        if gcb_config is None:
            gcb_config = dict(layers=[False]*len(layers))

        self.norm = norm
        self.input_channels = 128
        self.out_channels = 512

        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=3,
            pad_mode="pad",
            padding=1,
        )
        self.bn1 = norm(64)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            pad_mode="pad",
            padding=1,
        )
        self.bn2 = norm(128)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")

        self.layer1 = self._make_layer(
            block,
            256,
            layers[0],
            use_gcb=gcb_config["layers"][0],
            gcb_config=gcb_config,
        )
        self.conv3 = nn.Conv2d(
            256,
            256,
            kernel_size=3,
            pad_mode="pad",
            padding=1,
        )
        self.bn3 = nn.BatchNorm2d(256)

        self.layer2 = self._make_layer(
            block,
            256,
            layers[1],
            use_gcb=gcb_config["layers"][1],
            gcb_config=gcb_config,
        )
        self.conv4 = nn.Conv2d(
            256,
            256,
            kernel_size=3,
            pad_mode="pad",
            padding=1,
        )
        self.bn4 = nn.BatchNorm2d(256)

        self.max_pool_2 = nn.MaxPool2d(
            kernel_size=(2, 1), stride=(2, 1), pad_mode="same"
        )

        self.layer3 = self._make_layer(
            block,
            512,
            layers[2],
            use_gcb=gcb_config["layers"][2],
            gcb_config=gcb_config,
        )
        self.conv5 = nn.Conv2d(
            512,
            512,
            kernel_size=3,
            pad_mode="pad",
            padding=1,
        )
        self.bn5 = nn.BatchNorm2d(512)

        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            use_gcb=gcb_config["layers"][3],
            gcb_config=gcb_config,
        )
        self.conv6 = nn.Conv2d(
            512,
            512,
            kernel_size=3,
            pad_mode="pad",
            padding=1,
        )
        self.bn6 = nn.BatchNorm2d(512)

    def _make_layer(
        self,
        block: Type[BasicBlockGCAtten],
        channels: int,
        block_nums: int,
        stride: int = 1,
        use_gcb: bool = False,
        gcb_config: Optional[Dict[str, Any]] = None,
    ) -> nn.SequentialCell:
        """build model depending on cfgs"""
        down_sample = None

        if stride != 1 or self.input_channels != channels * block.expansion:
            down_sample = nn.SequentialCell(
                [
                    nn.Conv2d(
                        self.input_channels,
                        channels * block.expansion,
                        kernel_size=1,
                        stride=stride,
                    ),
                    nn.BatchNorm2d(channels * block.expansion),
                ]
            )

        layers = []
        layers.append(
            block(
                self.input_channels,
                channels,
                stride=stride,
                down_sample=down_sample,
                norm=self.norm,
                use_gcb=use_gcb,
                gcb_config=gcb_config,
            )
        )
        self.input_channels = channels * block.expansion

        for _ in range(1, block_nums):
            layers.append(
                block(
                    self.input_channels,
                    channels,
                )
            )

        return nn.SequentialCell(layers)

    def forward_features(self, x: Tensor) -> Tensor:
        """Network forward feature extraction."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.max_pool_2(x)

        x = self.layer3(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)

        return x

    def construct(self, x: Tensor) -> List[Tensor]:
        x = self.forward_features(x)
        return [x]


@register_backbone
def rec_resnet_master_resnet31(pretrained: bool = False, **kwargs: Any) -> RecResNetMaster:
    """Create the MASTER model with Resnet 31 layers.
    Args:
        pretrained: Use the pretrained weight. Default: False
        **kwargs: Dummy arguments for compatibility only.
    """
    gcb_config = {
        "ratio": 0.0625,
        "headers": 1,
        "att_scale": False,
        "fusion_type": "channel_add",
        "layers": [False, True, True, True],
    }

    model = RecResNetMaster(
        BasicBlockGCAtten,
        layers=[1, 2, 5, 3],
        in_channels=3,
        gcb_config=gcb_config,
    )

    # load pretrained weights
    if pretrained is True:
        raise NotImplementedError(
            "The default pretrained checkpoint for `rec_resnet_master_resnet31` backbone does not exist."
        )

    return model
