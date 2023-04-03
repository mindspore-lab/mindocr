"""
MindSpore implementation of `Visformer`.
Refer to: Visformer: The Vision-friendly Transformer
"""

from typing import List

import numpy as np

import mindspore
from mindspore import Tensor, nn, ops
from mindspore.common.initializer import Constant, HeNormal, TruncatedNormal, initializer

from .layers import DropPath, GlobalAvgPooling, Identity
from .registry import register_model
from .utils import _ntuple, load_pretrained

__all__ = [
    "Visformer",
    "visformer_tiny",
    "visformer_small",
    "visformer_tiny_v2",
    "visformer_small_v2",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "",
        "classifier": "",
        **kwargs,
    }


default_cfgs = {
    "visformer_small": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/visformer/visformer_small-6c83b6db.ckpt"),
    "visformer_tiny": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/visformer/visformer_tiny-daee0322.ckpt"),
    "visformer_tiny_v2": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/visformer/visformer_tiny_v2-6711a758.ckpt"),
    "visformer_small_v2": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/visformer/visformer_small_v2-63674ade.ckpt"),
}

to_2tuple = _ntuple(2)


class Mlp(nn.Cell):
    """MLP layer"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: nn.Cell = nn.GELU,
        drop: float = 0.0,
        group: int = 8,
        spatial_conv: bool = False,
    ) -> None:
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.out_features = out_features
        self.spatial_conv = spatial_conv
        if self.spatial_conv:
            if group < 2:
                hidden_features = in_features * 5 // 6
            else:
                hidden_features = in_features * 2
        self.hidden_features = hidden_features
        self.group = group
        self.drop = nn.Dropout(1 - drop)
        self.conv1 = nn.Conv2d(in_features, hidden_features, 1, 1, pad_mode="pad", padding=0)
        self.act1 = act_layer()
        if self.spatial_conv:
            self.conv2 = nn.Conv2d(hidden_features, hidden_features, 3, 1, pad_mode="pad", padding=1, group=self.group)
            self.act2 = act_layer()
        self.conv3 = nn.Conv2d(hidden_features, out_features, 1, 1, pad_mode="pad", padding=0)

    def construct(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.act1(x)
        x = self.drop(x)

        if self.spatial_conv:
            x = self.conv2(x)
            x = self.act2(x)

        x = self.conv3(x)
        x = self.drop(x)
        return x


class Attention(nn.Cell):
    """Attention layer"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim_ratio: float = 1.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super(Attention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = round(dim // num_heads * head_dim_ratio)
        self.head_dim = head_dim

        qk_scale_factor = qk_scale if qk_scale is not None else -0.25
        self.scale = head_dim**qk_scale_factor

        self.qkv = nn.Conv2d(dim, head_dim * num_heads * 3, 1, 1, pad_mode="pad", padding=0, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(1 - attn_drop)
        self.proj = nn.Conv2d(self.head_dim * self.num_heads, dim, 1, 1, pad_mode="pad", padding=0)
        self.proj_drop = nn.Dropout(1 - proj_drop)

    def construct(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x = self.qkv(x)
        qkv = ops.reshape(x, (B, 3, self.num_heads, self.head_dim, H * W))
        qkv = qkv.transpose((1, 0, 2, 4, 3))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ops.matmul(q * self.scale, k.transpose(0, 1, 3, 2) * self.scale)
        attn = ops.Softmax(axis=-1)(attn)
        attn = self.attn_drop(attn)
        x = ops.matmul(attn, v)

        x = x.transpose((0, 1, 3, 2)).reshape((B, -1, H, W))
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Cell):
    """visformer basic block"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim_ratio: float = 1.0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Cell = nn.GELU,
        group: int = 8,
        attn_disabled: bool = False,
        spatial_conv: bool = False,
    ) -> None:
        super(Block, self).__init__()
        self.attn_disabled = attn_disabled
        self.spatial_conv = spatial_conv
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        if not attn_disabled:
            self.norm1 = nn.BatchNorm2d(dim)
            self.attn = Attention(dim, num_heads=num_heads, head_dim_ratio=head_dim_ratio, qkv_bias=qkv_bias,
                                  qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       group=group, spatial_conv=spatial_conv)

    def construct(self, x: Tensor) -> Tensor:
        if not self.attn_disabled:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Cell):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, pad_mode="pad", padding=0,
                              has_bias=True)
        self.norm = nn.BatchNorm2d(embed_dim)

    def construct(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return x


class Visformer(nn.Cell):
    r"""Visformer model class, based on
    '"Visformer: The Vision-friendly Transformer"
    <https://arxiv.org/pdf/2104.12533.pdf>'

    Args:
        image_size (int) : images input size. Default: 224.
        number the channels of the input. Default: 32.
        num_classes (int) : number of classification classes. Default: 1000.
        embed_dim (int) : embedding dimension in all head. Default: 384.
        depth (int) : model block depth. Default: None.
        num_heads (int) : number of heads. Default: None.
        mlp_ratio (float) : ratio of hidden features in Mlp. Default: 4.
        qkv_bias (bool) : have bias in qkv layers or not. Default: False.
        qk_scale (float) : Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float) : dropout rate. Default: 0.
        attn_drop_rate (float) : attention layers dropout rate. Default: 0.
        drop_path_rate (float) : drop path rate. Default: 0.1.
        attn_stage (str) : block will have a attention layer if value = '1' else not. Default: '1111'.
        pos_embed (bool) : position embedding. Default: True.
        spatial_conv (str) : block will have a spatial convolution layer if value = '1' else not. Default: '1111'.
        group (int) : convolution group. Default: 8.
        pool (bool) : if true will use global_pooling else not. Default: True.
        conv_init : if true will init convolution weights else not. Default: False.
    """

    def __init__(
        self,
        img_size: int = 224,
        init_channels: int = 32,
        num_classes: int = 1000,
        embed_dim: int = 384,
        depth: List[int] = None,
        num_heads: List[int] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        attn_stage: str = "1111",
        pos_embed: bool = True,
        spatial_conv: str = "1111",
        group: int = 8,
        pool: bool = True,
        conv_init: bool = False,
    ) -> None:
        super(Visformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.init_channels = init_channels
        self.img_size = img_size
        self.pool = pool
        self.conv_init = conv_init
        self.depth = depth
        assert (isinstance(depth, list) or isinstance(depth, tuple)) and len(depth) == 4
        if not (isinstance(num_heads, list) or isinstance(num_heads, tuple)):
            num_heads = [num_heads] * 4

        self.pos_embed = pos_embed
        dpr = np.linspace(0, drop_path_rate, sum(depth)).tolist()

        self.stem = nn.SequentialCell([
            nn.Conv2d(3, self.init_channels, 7, 2, pad_mode="pad", padding=3),
            nn.BatchNorm2d(self.init_channels),
            nn.ReLU()
        ])
        img_size //= 2

        self.pos_drop = nn.Dropout(1 - drop_rate)
        # stage0
        if depth[0]:
            self.patch_embed0 = PatchEmbed(img_size=img_size, patch_size=2, in_chans=self.init_channels,
                                           embed_dim=embed_dim // 4)
            img_size //= 2
            if self.pos_embed:
                self.pos_embed0 = mindspore.Parameter(
                    ops.zeros((1, embed_dim // 4, img_size, img_size), mindspore.float32))
            self.stage0 = nn.CellList([
                Block(dim=embed_dim // 4, num_heads=num_heads[0], head_dim_ratio=0.25, mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                      group=group, attn_disabled=(attn_stage[0] == "0"), spatial_conv=(spatial_conv[0] == "1"))
                for i in range(depth[0])
            ])

        # stage1
        if depth[0]:
            self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=2, in_chans=embed_dim // 4,
                                           embed_dim=embed_dim // 2)
            img_size //= 2
        else:
            self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=4, in_chans=self.init_channels,
                                           embed_dim=embed_dim // 2)
            img_size //= 4

        if self.pos_embed:
            self.pos_embed1 = mindspore.Parameter(ops.zeros((1, embed_dim // 2, img_size, img_size), mindspore.float32))

        self.stage1 = nn.CellList([
            Block(
                dim=embed_dim // 2, num_heads=num_heads[1], head_dim_ratio=0.5, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                group=group, attn_disabled=(attn_stage[1] == "0"), spatial_conv=(spatial_conv[1] == "1")
            )
            for i in range(sum(depth[:1]), sum(depth[:2]))
        ])

        # stage2
        self.patch_embed2 = PatchEmbed(img_size=img_size, patch_size=2, in_chans=embed_dim // 2, embed_dim=embed_dim)
        img_size //= 2
        if self.pos_embed:
            self.pos_embed2 = mindspore.Parameter(ops.zeros((1, embed_dim, img_size, img_size), mindspore.float32))
        self.stage2 = nn.CellList([
            Block(
                dim=embed_dim, num_heads=num_heads[2], head_dim_ratio=1.0, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                group=group, attn_disabled=(attn_stage[2] == "0"), spatial_conv=(spatial_conv[2] == "1")
            )
            for i in range(sum(depth[:2]), sum(depth[:3]))
        ])

        # stage3
        self.patch_embed3 = PatchEmbed(img_size=img_size, patch_size=2, in_chans=embed_dim, embed_dim=embed_dim * 2)
        img_size //= 2
        if self.pos_embed:
            self.pos_embed3 = mindspore.Parameter(ops.zeros((1, embed_dim * 2, img_size, img_size), mindspore.float32))
        self.stage3 = nn.CellList([
            Block(
                dim=embed_dim * 2, num_heads=num_heads[3], head_dim_ratio=1.0, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                group=group, attn_disabled=(attn_stage[3] == "0"), spatial_conv=(spatial_conv[3] == "1")
            )
            for i in range(sum(depth[:3]), sum(depth[:4]))
        ])

        # head
        if self.pool:
            self.global_pooling = GlobalAvgPooling()

        self.norm = nn.BatchNorm2d(embed_dim * 2)
        self.head = nn.Dense(embed_dim * 2, num_classes)

        # weight init
        if self.pos_embed:
            if depth[0]:
                self.pos_embed0.set_data(initializer(TruncatedNormal(0.02),
                                                     self.pos_embed0.shape, self.pos_embed0.dtype))
            self.pos_embed1.set_data(initializer(TruncatedNormal(0.02),
                                                 self.pos_embed1.shape, self.pos_embed1.dtype))
            self.pos_embed2.set_data(initializer(TruncatedNormal(0.02),
                                                 self.pos_embed2.shape, self.pos_embed2.dtype))
            self.pos_embed3.set_data(initializer(TruncatedNormal(0.02),
                                                 self.pos_embed3.shape, self.pos_embed3.dtype))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer(TruncatedNormal(0.02), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer(Constant(0), cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.beta.set_data(initializer(Constant(0), cell.beta.shape, cell.beta.dtype))
                cell.gamma.set_data(initializer(Constant(1), cell.gamma.shape, cell.gamma.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.beta.set_data(initializer(Constant(0), cell.beta.shape, cell.beta.dtype))
                cell.gamma.set_data(initializer(Constant(1), cell.gamma.shape, cell.gamma.dtype))
            elif isinstance(cell, nn.Conv2d):
                if self.conv_init:
                    cell.weight.set_data(initializer(HeNormal(mode="fan_out", nonlinearity="relu"), cell.weight.shape,
                                                     cell.weight.dtype))
                else:
                    cell.weight.set_data(initializer(TruncatedNormal(0.02), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer(Constant(0), cell.bias.shape, cell.bias.dtype))

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.stem(x)

        # stage 0
        if self.depth[0]:
            x = self.patch_embed0(x)
            if self.pos_embed:
                x = x + self.pos_embed0
                x = self.pos_drop(x)
            for b in self.stage0:
                x = b(x)

        # stage 1
        x = self.patch_embed1(x)
        if self.pos_embed:
            x = x + self.pos_embed1
            x = self.pos_drop(x)
        for b in self.stage1:
            x = b(x)

        # stage 2
        x = self.patch_embed2(x)
        if self.pos_embed:
            x = x + self.pos_embed2
            x = self.pos_drop(x)
        for b in self.stage2:
            x = b(x)

        # stage 3
        x = self.patch_embed3(x)
        if self.pos_embed:
            x = x + self.pos_embed3
            x = self.pos_drop(x)
        for b in self.stage3:
            x = b(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        # head
        if self.pool:
            x = self.global_pooling(x)
        else:
            x = x[:, :, 0, 0]
        x = self.head(x.view(x.shape[0], -1))
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def visformer_tiny(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs):
    """Get visformer tiny model.
    Refer to the base class 'models.visformer' for more details.
    """
    default_cfg = default_cfgs["visformer_tiny"]
    model = Visformer(img_size=224, init_channels=16, num_classes=num_classes, embed_dim=192,
                      depth=[0, 7, 4, 4], num_heads=[3, 3, 3, 3], mlp_ratio=4., group=8,
                      attn_stage="0011", spatial_conv="1100", drop_path_rate=0.03, conv_init=True, **kwargs)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def visformer_tiny_v2(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs):
    """Get visformer tiny2 model.
    Refer to the base class 'models.visformer' for more details.
    """
    default_cfg = default_cfgs["visformer_tiny_v2"]
    model = Visformer(img_size=224, init_channels=24, num_classes=num_classes, embed_dim=192,
                      depth=[1, 4, 6, 3], num_heads=[1, 3, 6, 12], mlp_ratio=4., qk_scale=-0.5, group=8,
                      attn_stage="0011", spatial_conv="1100", drop_path_rate=0.03, conv_init=True, **kwargs)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def visformer_small(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs):
    """Get visformer small model.
    Refer to the base class 'models.visformer' for more details.
    """
    default_cfg = default_cfgs["visformer_small"]
    model = Visformer(img_size=224, init_channels=32, num_classes=num_classes, embed_dim=384,
                      depth=[0, 7, 4, 4], num_heads=[6, 6, 6, 6], mlp_ratio=4., group=8,
                      attn_stage="0011", spatial_conv="1100", conv_init=True, **kwargs)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def visformer_small_v2(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs):
    """Get visformer small2 model.
    Refer to the base class 'models.visformer' for more details.
    """
    default_cfg = default_cfgs["visformer_small_v2"]
    model = Visformer(img_size=224, init_channels=32, num_classes=num_classes, embed_dim=256,
                      depth=[1, 10, 14, 3], num_heads=[2, 4, 8, 16], mlp_ratio=4., qk_scale=-0.5,
                      group=8, attn_stage="0011", spatial_conv="1100", conv_init=True, **kwargs)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model
