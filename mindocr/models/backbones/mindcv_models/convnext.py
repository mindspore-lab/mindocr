"""
MindSpore implementation of `ConvNeXt`.
Refer to: A ConvNet for the 2020s
"""
from typing import List, Tuple

import numpy as np

import mindspore.common.initializer as init
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import nn, ops

from .layers.drop_path import DropPath
from .layers.identity import Identity
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "ConvNeXt",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "convnext_xlarge",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "feature.0.0",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "convnext_tiny": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/convnext/convnext_tiny-ae5ff8d7.ckpt"),
    "convnext_small": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/convnext/convnext_small-e23008f3.ckpt"),
    "convnext_base": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/convnext/convnext_base-ee3544b8.ckpt"),
    "convnext_large": _cfg(url=""),
    "convnext_xlarge": _cfg(url=""),
}


class ConvNextLayerNorm(nn.LayerNorm):
    r"""LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W)."""

    def __init__(
        self,
        normalized_shape: Tuple[int],
        epsilon: float,
        norm_axis: int = -1,
    ) -> None:
        super().__init__(normalized_shape=normalized_shape, epsilon=epsilon)
        assert norm_axis in (-1, 1), "ConvNextLayerNorm's norm_axis must be 1 or -1."
        self.norm_axis = norm_axis

    def construct(self, input_x: Tensor) -> Tensor:
        if self.norm_axis == -1:
            y, _, _ = self.layer_norm(input_x, self.gamma, self.beta)
        else:
            input_x = ops.transpose(input_x, (0, 2, 3, 1))
            y, _, _ = self.layer_norm(input_x, self.gamma, self.beta)
            y = ops.transpose(y, (0, 3, 1, 2))
        return y


class Block(nn.Cell):
    """ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, group=dim, has_bias=True)  # depthwise conv
        self.norm = ConvNextLayerNorm((dim,), epsilon=1e-6)
        self.pwconv1 = nn.Dense(dim, 4 * dim)  # pointwise/1x1 convs, implemented with Dense layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Dense(4 * dim, dim)
        self.gamma_ = Parameter(Tensor(layer_scale_init_value * np.ones((dim)), dtype=mstype.float32),
                                requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()

    def construct(self, x: Tensor) -> Tensor:
        downsample = x
        x = self.dwconv(x)
        x = ops.transpose(x, (0, 2, 3, 1))
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma_ is not None:
            x = self.gamma_ * x
        x = ops.transpose(x, (0, 3, 1, 2))
        x = downsample + self.drop_path(x)
        return x


class ConvNeXt(nn.Cell):
    r"""ConvNeXt model class, based on
    '"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>'
    Args:
        in_channels (int) : dim of the input channel.
        num_classes (int) : dim of the classes predicted.
        depths (List[int]) : the depths of each layer.
        dims (List[int]) : the middle dim of each layer.
        drop_path_rate (float) : the rate of droppath default : 0.
        layer_scale_init_value (float) : the parameter of init for the classifier default : 1e-6.
        head_init_scale (float) : the parameter of init for the head default : 1.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        depths: List[int],
        dims: List[int],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
    ):
        super().__init__()

        self.downsample_layers = nn.CellList()  # stem and 3 intermediate down_sampling conv layers
        stem = nn.SequentialCell(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4, has_bias=True),
            ConvNextLayerNorm((dims[0],), epsilon=1e-6, norm_axis=1),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.SequentialCell(
                ConvNextLayerNorm((dims[i],), epsilon=1e-6, norm_axis=1),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2, has_bias=True),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.CellList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = list(np.linspace(0, drop_path_rate, sum(depths)))
        cur = 0
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                blocks.append(Block(dim=dims[i], drop_path=dp_rates[cur + j],
                                    layer_scale_init_value=layer_scale_init_value))
            stage = nn.SequentialCell(blocks)
            self.stages.append(stage)
            cur += depths[i]

        self.norm = ConvNextLayerNorm((dims[-1],), epsilon=1e-6)  # final norm layer
        self.classifier = nn.Dense(dims[-1], num_classes)  # classifier
        self.feature = nn.SequentialCell([
            self.downsample_layers[0],
            self.stages[0],
            self.downsample_layers[1],
            self.stages[1],
            self.downsample_layers[2],
            self.stages[2],
            self.downsample_layers[3],
            self.stages[3]
        ])
        self.head_init_scale = head_init_scale
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, (nn.Dense, nn.Conv2d)):
                cell.weight.set_data(
                    init.initializer(init.TruncatedNormal(sigma=0.02), cell.weight.shape, cell.weight.dtype)
                )
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(init.initializer(init.Zero(), cell.bias.shape, cell.bias.dtype))
        self.classifier.weight.set_data(self.classifier.weight * self.head_init_scale)
        self.classifier.bias.set_data(self.classifier.bias * self.head_init_scale)

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.classifier(x)
        return x

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.feature(x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def convnext_tiny(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    """Get ConvNeXt tiny model.
    Refer to the base class 'models.ConvNeXt' for more details.
    """
    default_cfg = default_cfgs["convnext_tiny"]
    model = ConvNeXt(
        in_channels=in_channels, num_classes=num_classes, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs
    )

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def convnext_small(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    """Get ConvNeXt small model.
    Refer to the base class 'models.ConvNeXt' for more details.
    """
    default_cfg = default_cfgs["convnext_small"]
    model = ConvNeXt(
        in_channels=in_channels, num_classes=num_classes, depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs
    )

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def convnext_base(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    """Get ConvNeXt base model.
    Refer to the base class 'models.ConvNeXt' for more details.
    """
    default_cfg = default_cfgs["convnext_base"]
    model = ConvNeXt(
        in_channels=in_channels, num_classes=num_classes, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs
    )

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def convnext_large(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    """Get ConvNeXt large model.
    Refer to the base class 'models.ConvNeXt' for more details.
    """
    default_cfg = default_cfgs["convnext_large"]
    model = ConvNeXt(
        in_channels=in_channels, num_classes=num_classes, depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs
    )

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def convnext_xlarge(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    """Get ConvNeXt xlarge model.
    Refer to the base class 'models.ConvNeXt' for more details.
    """
    default_cfg = default_cfgs["convnext_xlarge"]
    model = ConvNeXt(
        in_channels=in_channels, num_classes=num_classes, depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs
    )

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
