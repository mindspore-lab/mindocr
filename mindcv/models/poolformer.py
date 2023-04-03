"""
MindSpore implementation of `poolformer`.
Refer to PoolFormer: MetaFormer Is Actually What You Need for Vision.
"""

import collections.abc
from itertools import repeat

import numpy as np

import mindspore
import mindspore.common.initializer as init
from mindspore import Tensor, nn, ops

from .layers import DropPath, Identity
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "PoolFormer",
    "poolformer_s12",
    "poolformer_s24",
    "poolformer_s36",
    "poolformer_m36",
    "poolformer_m48",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "",
        "classifier": "",
        **kwargs,
    }


default_cfgs = dict(
    poolformer_s12=_cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/poolformer/poolformer_s12-5be5c4e4.ckpt", crop_pct=0.9
    ),
    poolformer_s24=_cfg(url="", crop_pct=0.9),
    poolformer_s36=_cfg(url="", crop_pct=0.9),
    poolformer_m36=_cfg(url="", crop_pct=0.95),
    poolformer_m48=_cfg(url="", crop_pct=0.95),
)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class ConvMlp(nn.Cell):
    """MLP using 1x1 convs that keeps spatial dims"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, has_bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else Identity()
        self.act = act_layer(approximate=False)
        self.drop = nn.Dropout(1 - drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, has_bias=bias[1])
        self.cls_init_weights()

    def cls_init_weights(self):
        """Initialize weights for cells."""
        for name, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                m.weight.set_data(
                    init.initializer(init.TruncatedNormal(sigma=.02), m.weight.shape, m.weight.dtype))
                if m.bias is not None:
                    m.bias.set_data(
                        init.initializer(init.Constant(0), m.bias.shape, m.bias.dtype))

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Cell):
    """Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]"""

    def __init__(self, in_chs=3, embed_dim=768, patch_size=16, stride=16, padding=0, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        # padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chs, embed_dim, kernel_size=patch_size, stride=stride, padding=padding, pad_mode="pad",
                              has_bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer else Identity()

    def construct(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class Pooling(nn.Cell):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, pad_mode="same")

    def construct(self, x):
        return self.pool(x) - x


class PoolFormerBlock(nn.Cell):
    """Implementation of one PoolFormer block."""

    def __init__(
        self,
        dim,
        pool_size=3,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer=nn.GroupNorm,
        drop=0.0,
        drop_path=0.0,
        layer_scale_init_value=1e-5,
    ):
        super().__init__()
        self.norm1 = norm_layer(1, dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(1, dim)
        self.mlp = ConvMlp(dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if layer_scale_init_value:
            layer_scale_init_tensor = Tensor(layer_scale_init_value * np.ones([dim]).astype(np.float32))
            self.layer_scale_1 = mindspore.Parameter(layer_scale_init_tensor)
            self.layer_scale_2 = mindspore.Parameter(layer_scale_init_tensor)
        else:
            self.layer_scale_1 = None
            self.layer_scale_2 = None
        self.expand_dims = ops.ExpandDims()

    def construct(self, x):
        if self.layer_scale_1 is not None:
            x = x + self.drop_path(
                self.expand_dims(self.expand_dims(self.layer_scale_1, -1), -1) * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.expand_dims(self.expand_dims(self.layer_scale_2, -1), -1) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(
    dim,
    index,
    layers,
    pool_size=3,
    mlp_ratio=4.0,
    act_layer=nn.GELU,
    norm_layer=nn.GroupNorm,
    drop_rate=0.0,
    drop_path_rate=0.0,
    layer_scale_init_value=1e-5,
):
    """generate PoolFormer blocks for a stage"""
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PoolFormerBlock(
            dim, pool_size=pool_size, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop=drop_rate, drop_path=block_dpr,
            layer_scale_init_value=layer_scale_init_value,
        ))
    blocks = nn.SequentialCell(*blocks)
    return blocks


class PoolFormer(nn.Cell):
    r"""PoolFormer model class, based on
    `"MetaFormer Is Actually What You Need for Vision" <https://arxiv.org/pdf/2111.11418v3.pdf>`_

    Args:
        layers: number of blocks for the 4 stages
        embed_dims: the embedding dims for the 4 stages. Default: (64, 128, 320, 512)
        mlp_ratios: mlp ratios for the 4 stages. Default: (4, 4, 4, 4)
        downsamples: flags to apply downsampling or not. Default: (True, True, True, True)
        pool_size: the pooling size for the 4 stages. Default: 3
        in_chans: number of input channels. Default: 3
        num_classes: number of classes for the image classification. Default: 1000
        global_pool: define the types of pooling layer. Default: avg
        norm_layer: define the types of normalization. Default: nn.GroupNorm
        act_layer: define the types of activation. Default: nn.GELU
        in_patch_size: specify the patch embedding for the input image. Default: 7
        in_stride: specify the stride for the input image. Default: 4.
        in_pad: specify the pad for the input image. Default: 2.
        down_patch_size: specify the downsample. Default: 3.
        down_stride: specify the downsample (patch embed.). Default: 2.
        down_pad: specify the downsample (patch embed.). Default: 1.
        drop_rate: dropout rate of the layer before main classifier. Default: 0.
        drop_path_rate: Stochastic Depth. Default: 0.
        layer_scale_init_value: LayerScale. Default: 1e-5.
        fork_feat: whether output features of the 4 stages, for dense prediction. Default: False.
    """

    def __init__(
        self,
        layers,
        embed_dims=(64, 128, 320, 512),
        mlp_ratios=(4, 4, 4, 4),
        downsamples=(True, True, True, True),
        pool_size=3,
        in_chans=3,
        num_classes=1000,
        global_pool="avg",
        norm_layer=nn.GroupNorm,
        act_layer=nn.GELU,
        in_patch_size=7,
        in_stride=4,
        in_pad=2,
        down_patch_size=3,
        down_stride=2,
        down_pad=1,
        drop_rate=0.0,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-5,
        fork_feat=False,
    ):
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.global_pool = global_pool
        self.num_features = embed_dims[-1]
        self.grad_checkpointing = False

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chs=in_chans, embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            network.append(basic_blocks(
                embed_dims[i], i, layers,
                pool_size=pool_size, mlp_ratio=mlp_ratios[i],
                act_layer=act_layer, norm_layer=norm_layer,
                drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value)
            )
            if i < len(layers) - 1 and (downsamples[i] or embed_dims[i] != embed_dims[i + 1]):
                # downsampling between stages
                network.append(PatchEmbed(
                    in_chs=embed_dims[i], embed_dim=embed_dims[i + 1],
                    patch_size=down_patch_size, stride=down_stride, padding=down_pad)
                )

        self.network = nn.SequentialCell(*network)
        self.norm = norm_layer(1, embed_dims[-1])
        self.head = nn.Dense(embed_dims[-1], num_classes, has_bias=True) if num_classes > 0 else Identity()
        # self._initialize_weights()
        self.cls_init_weights()

    def cls_init_weights(self):
        """Initialize weights for cells."""
        for name, m in self.cells_and_names():
            if isinstance(m, nn.Dense):
                m.weight.set_data(
                    init.initializer(init.TruncatedNormal(sigma=.02), m.weight.shape, m.weight.dtype))
                if m.bias is not None:
                    m.bias.set_data(
                        init.initializer(init.Constant(0), m.bias.shape, m.bias.dtype))

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Dense(self.num_features, num_classes) if num_classes > 0 else Identity()

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x = self.network(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        return self.head(x.mean([-2, -1]))

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        return self.forward_head(x)


@register_model
def poolformer_s12(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> PoolFormer:
    """Get poolformer_s12 model.
    Refer to the base class `models.PoolFormer` for more details."""
    default_cfg = default_cfgs["poolformer_s12"]
    model = PoolFormer(in_chans=in_channels, num_classes=num_classes, layers=(2, 2, 6, 2), **kwargs)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def poolformer_s24(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> PoolFormer:
    """Get poolformer_s24 model.
    Refer to the base class `models.PoolFormer` for more details."""
    default_cfg = default_cfgs["poolformer_s24"]
    model = PoolFormer(in_chans=in_channels, num_classes=num_classes, layers=(4, 4, 12, 4), **kwargs)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def poolformer_s36(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> PoolFormer:
    """Get poolformer_s36 model.
    Refer to the base class `models.PoolFormer` for more details."""
    default_cfg = default_cfgs["poolformer_s36"]
    model = PoolFormer(
        in_chans=in_channels, num_classes=num_classes, layers=(6, 6, 18, 6), layer_scale_init_value=1e-6, **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def poolformer_m36(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> PoolFormer:
    """Get poolformer_m36 model.
    Refer to the base class `models.PoolFormer` for more details."""
    default_cfg = default_cfgs["poolformer_m36"]
    layers = (6, 6, 18, 6)
    embed_dims = (96, 192, 384, 768)
    model = PoolFormer(
        in_chans=in_channels,
        num_classes=num_classes,
        layers=layers,
        layer_scale_init_value=1e-6,
        embed_dims=embed_dims,
        **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def poolformer_m48(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> PoolFormer:
    """Get poolformer_m48 model.
    Refer to the base class `models.PoolFormer` for more details."""
    default_cfg = default_cfgs["poolformer_m48"]
    layers = (8, 8, 24, 8)
    embed_dims = (96, 192, 384, 768)
    model = PoolFormer(
        in_chans=in_channels,
        num_classes=num_classes,
        layers=layers,
        layer_scale_init_value=1e-6,
        embed_dims=embed_dims,
        **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model
