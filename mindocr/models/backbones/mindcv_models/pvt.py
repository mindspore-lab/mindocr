"""
MindSpore implementation of `PVT`.
Refer to PVT: Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions
"""
import math
from functools import partial
from typing import Optional

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common import initializer as weight_init

from .layers.drop_path import DropPath
from .layers.identity import Identity
from .layers.mlp import Mlp
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "PyramidVisionTransformer",
    "pvt_tiny",
    "pvt_small",
    "pvt_medium",
    "pvt_large",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "patch_embed1.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "pvt_tiny": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/pvt/pvt_tiny-6abb953d.ckpt"),
    "pvt_small": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/pvt/pvt_small-213c2ed1.ckpt"),
    "pvt_medium": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/pvt/pvt_medium-469e6802.ckpt"),
    "pvt_large": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/pvt/pvt_large-bb6895d7.ckpt"),
}


class Attention(nn.Cell):
    """spatial-reduction attention (SRA)"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
    ):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.kv = nn.Dense(dim, dim * 2, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(1 - attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(1 - proj_drop)
        self.qk_batmatmul = ops.BatchMatMul(transpose_b=True)
        self.batmatmul = ops.BatchMatMul()
        self.softmax = nn.Softmax(axis=-1)
        self.reshape = ops.reshape
        self.transpose = ops.transpose

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, has_bias=True)
            self.norm = nn.LayerNorm([dim])

    def construct(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x)
        q = self.reshape(q, (B, N, self.num_heads, C // self.num_heads))
        q = self.transpose(q, (0, 2, 1, 3))
        if self.sr_ratio > 1:
            x_ = self.reshape(self.transpose(x, (0, 2, 1)), (B, C, H, W))

            x_ = self.transpose(self.reshape(self.sr(x_), (B, C, -1)), (0, 2, 1))
            x_ = self.norm(x_)
            kv = self.kv(x_)

            kv = self.transpose(self.reshape(kv, (B, -1, 2, self.num_heads, C // self.num_heads)), (2, 0, 3, 1, 4))
        else:
            kv = self.kv(x)
            kv = self.transpose(self.reshape(kv, (B, -1, 2, self.num_heads, C // self.num_heads)), (2, 0, 3, 1, 4))
        k, v = kv[0], kv[1]
        attn = self.qk_batmatmul(q, k) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = self.batmatmul(attn, v)
        x = self.reshape(self.transpose(x, (0, 2, 1, 3)), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Cell):
    """ Block with spatial-reduction attention (SRA) and feed forward"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(Block, self).__init__()
        self.norm1 = norm_layer([dim], epsilon=1e-5)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer([dim])
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x, H, W):
        x1 = self.norm1(x)
        x1 = self.attn(x1, H, W)
        x = x + self.drop_path(x1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Cell):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size

        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=True)
        self.norm = nn.LayerNorm([embed_dim], epsilon=1e-5)
        self.reshape = ops.reshape
        self.transpose = ops.transpose

    def construct(self, x):
        B, C, H, W = x.shape

        x = self.proj(x)
        b, c, h, w = x.shape
        x = self.reshape(x, (b, c, h * w))
        x = self.transpose(x, (0, 2, 1))
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PyramidVisionTransformer(nn.Cell):
    r"""Pyramid Vision Transformer model class, based on
    `"Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions" <https://arxiv.org/abs/2102.12122>`_  # noqa: E501

    Args:
        img_size(int) : size of a input image.
        patch_size (int) : size of a single image patch.
        in_chans (int) : number the channels of the input. Default: 3.
        num_classes (int) : number of classification classes. Default: 1000.
        embed_dims (list) : how many hidden dim in each PatchEmbed.
        num_heads (list) : number of attention head in each stage.
        mlp_ratios (list): ratios of MLP hidden dims in each stage.
        qkv_bias(bool) : use bias in attention.
        qk_scale(float) : Scale multiplied by qk in attention(if not none), otherwise head_dim ** -0.5.
        drop_rate(float) : The drop rate for each block. Default: 0.0.
        attn_drop_rate(float) : The drop rate for attention. Default: 0.0.
        drop_path_rate(float) : The drop rate for drop path. Default: 0.0.
        norm_layer(nn.Cell) : Norm layer that will be used in blocks. Default: nn.LayerNorm.
        depths (list) : number of Blocks.
        sr_ratios(list) : stride and kernel size of each attention.
        num_stages(int) : number of stage. Default: 4.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], num_stages=4):
        super(PyramidVisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        start = Tensor(0, mindspore.float32)
        stop = Tensor(drop_path_rate, mindspore.float32)
        dpr = [float(x) for x in ops.linspace(start, stop, sum(depths))]  # stochastic depth decay rule
        cur = 0
        b_list = []
        self.pos_embed = []
        self.pos_drop = nn.Dropout(1 - drop_rate)
        for i in range(num_stages):
            block = nn.CellList(
                [Block(dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                       qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                       norm_layer=norm_layer, sr_ratio=sr_ratios[i])
                 for j in range(depths[i])
                 ])

            b_list.append(block)
            cur += depths[0]

        self.patch_embed1 = PatchEmbed(img_size=img_size,
                                       patch_size=patch_size,
                                       in_chans=in_chans,
                                       embed_dim=embed_dims[0])
        num_patches = self.patch_embed1.num_patches
        self.pos_embed1 = mindspore.Parameter(ops.zeros((1, num_patches, embed_dims[0]), mindspore.float16))
        self.pos_drop1 = nn.Dropout(1 - drop_rate)

        self.patch_embed2 = PatchEmbed(img_size=img_size // (2 ** (1 + 1)),
                                       patch_size=2,
                                       in_chans=embed_dims[1 - 1],
                                       embed_dim=embed_dims[1])
        num_patches = self.patch_embed2.num_patches
        self.pos_embed2 = mindspore.Parameter(ops.zeros((1, num_patches, embed_dims[1]), mindspore.float16))
        self.pos_drop2 = nn.Dropout(1 - drop_rate)

        self.patch_embed3 = PatchEmbed(img_size=img_size // (2 ** (2 + 1)),
                                       patch_size=2,
                                       in_chans=embed_dims[2 - 1],
                                       embed_dim=embed_dims[2])
        num_patches = self.patch_embed3.num_patches
        self.pos_embed3 = mindspore.Parameter(ops.zeros((1, num_patches, embed_dims[2]), mindspore.float16))
        self.pos_drop3 = nn.Dropout(1 - drop_rate)

        self.patch_embed4 = PatchEmbed(img_size // (2 ** (3 + 1)),
                                       patch_size=2,
                                       in_chans=embed_dims[3 - 1],
                                       embed_dim=embed_dims[3])
        num_patches = self.patch_embed4.num_patches + 1
        self.pos_embed4 = mindspore.Parameter(ops.zeros((1, num_patches, embed_dims[3]), mindspore.float16))
        self.pos_drop4 = nn.Dropout(1 - drop_rate)
        self.Blocks = nn.CellList(b_list)

        self.norm = norm_layer([embed_dims[3]])

        # cls_token
        self.cls_token = mindspore.Parameter(ops.zeros((1, 1, embed_dims[3]), mindspore.float32))

        # classification head
        self.head = nn.Dense(embed_dims[3], num_classes) if num_classes > 0 else Identity()
        self.reshape = ops.reshape
        self.transpose = ops.transpose
        self.tile = ops.Tile()
        self.Concat = ops.Concat(axis=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape, cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(), cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(), cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(), cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Conv2d):
                fan_out = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                fan_out //= cell.group
                cell.weight.set_data(weight_init.initializer(weight_init.Normal(sigma=math.sqrt(2.0 / fan_out)),
                                                             cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(), cell.bias.shape, cell.bias.dtype))

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Dense(self.embed_dim, num_classes) if num_classes > 0 else Identity()

    def _get_pos_embed(self, pos_embed, ph, pw, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            ResizeBilinear = nn.ResizeBilinear()

            pos_embed = self.transpose(self.reshape(pos_embed, (1, ph, pw, -1)), (0, 3, 1, 2))
            pos_embed = ResizeBilinear(pos_embed, (H, W))

            pos_embed = self.transpose(self.reshape(pos_embed, (1, -1, H * W)), (0, 2, 1))

            return pos_embed

    def forward_features(self, x):
        B = x.shape[0]

        x, (H, W) = self.patch_embed1(x)
        pos_embed = self.pos_embed1
        x = self.pos_drop1(x + pos_embed)
        for blk in self.Blocks[0]:
            x = blk(x, H, W)
        x = self.transpose(self.reshape(x, (B, H, W, -1)), (0, 3, 1, 2))

        x, (H, W) = self.patch_embed2(x)
        ph, pw = self.patch_embed2.H, self.patch_embed2.W
        pos_embed = self._get_pos_embed(self.pos_embed2, ph, pw, H, W)
        x = self.pos_drop2(x + pos_embed)
        for blk in self.Blocks[1]:
            x = blk(x, H, W)
        x = self.transpose(self.reshape(x, (B, H, W, -1)), (0, 3, 1, 2))

        x, (H, W) = self.patch_embed3(x)
        ph, pw = self.patch_embed3.H, self.patch_embed3.W
        pos_embed = self._get_pos_embed(self.pos_embed3, ph, pw, H, W)
        x = self.pos_drop3(x + pos_embed)
        for blk in self.Blocks[2]:
            x = blk(x, H, W)
        x = self.transpose(self.reshape(x, (B, H, W, -1)), (0, 3, 1, 2))

        x, (H, W) = self.patch_embed4(x)
        cls_tokens = self.tile(self.cls_token, (B, 1, 1))

        x = self.Concat((cls_tokens, x))
        ph, pw = self.patch_embed4.H, self.patch_embed4.W
        pos_embed_ = self._get_pos_embed(self.pos_embed4[:, 1:], ph, pw, H, W)
        pos_embed = self.Concat((self.pos_embed4[:, 0:1], pos_embed_))
        x = self.pos_drop4(x + pos_embed)
        for blk in self.Blocks[3]:
            x = blk(x, H, W)

        x = self.norm(x)

        return x[:, 0]

    def forward_head(self, x: Tensor) -> Tensor:
        return self.head(x)

    def construct(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)

        return x


@register_model
def pvt_tiny(
    pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs
) -> PyramidVisionTransformer:
    """Get PVT tiny model
    Refer to the base class "models.PVT" for more details.
    """
    default_cfg = default_cfgs['pvt_tiny']
    model = PyramidVisionTransformer(in_chans=in_channels, num_classes=num_classes,
                                     patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                                     mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                     norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[2, 2, 2, 2],
                                     sr_ratios=[8, 4, 2, 1], **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def pvt_small(
    pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs
) -> PyramidVisionTransformer:
    """Get PVT small model
    Refer to the base class "models.PVT" for more details.
    """
    default_cfg = default_cfgs['pvt_small']
    model = PyramidVisionTransformer(in_chans=in_channels, num_classes=num_classes,
                                     patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                                     mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                     norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 4, 6, 3],
                                     sr_ratios=[8, 4, 2, 1], **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def pvt_medium(
    pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs
) -> PyramidVisionTransformer:
    """Get PVT medium model
    Refer to the base class "models.PVT" for more details.
    """
    default_cfg = default_cfgs['pvt_medium']
    model = PyramidVisionTransformer(in_chans=in_channels, num_classes=num_classes,
                                     patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                                     mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                     norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 4, 18, 3],
                                     sr_ratios=[8, 4, 2, 1], **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def pvt_large(
    pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs
) -> PyramidVisionTransformer:
    """Get PVT large model
    Refer to the base class "models.PVT" for more details.
    """
    default_cfg = default_cfgs['pvt_large']
    model = PyramidVisionTransformer(in_chans=in_channels, num_classes=num_classes,
                                     patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                                     mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                     norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 8, 27, 3],
                                     sr_ratios=[8, 4, 2, 1], **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
