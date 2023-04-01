"""
MindSpore implementation of `CaiT`.
Refer to Going deeper with Image Transformers.
"""

from functools import partial

import mindspore as ms
import mindspore.common.initializer as init
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from mindspore.common.initializer import TruncatedNormal

from .layers.drop_path import DropPath
from .layers.mlp import Mlp
from .layers.patch_embed import PatchEmbed
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "CaiT",
    "cait_xxs24_224",
    "cait_xs24_384",
    "cait_s24_224",
    "cait_s24_384",
    "cait_s36_384",
    "cait_m36_384",
    "cait_m48_448",
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': '', 'classifier': '',
        **kwargs
    }


default_cfgs = {
    "cait_xxs24_224": _cfg(url=''),
    "cait_xs24_384": _cfg(url=''),
    "cait_s24_224": _cfg(url=''),
    "cait_s24_384": _cfg(url=''),
    "cait_s36_384": _cfg(url=''),
    "cait_m36_384": _cfg(url=''),
    "cait_m48_448": _cfg(url=''),
}


class ClassAttention(nn.Cell):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 attn_drop_rate: float = 0.,
                 proj_drop_rate: float = 0.) -> None:
        super(ClassAttention, self).__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads."
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.k = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.v = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(1 - attn_drop_rate)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(1 - proj_drop_rate)
        self.softmax = nn.Softmax(axis=-1)

        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)

    def construct(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        q = ops.expand_dims(self.q(x[:, 0]), 1)
        q = ops.reshape(q, (B, 1, self.num_heads, C // self.num_heads))
        q = ops.transpose(q, (0, 2, 1, 3))
        q = ops.mul(q, self.scale)
        k = ops.transpose(ops.reshape(self.k(x), (B, N, self.num_heads, C // self.num_heads)), (0, 2, 1, 3))
        v = ops.transpose(ops.reshape(self.v(x), (B, N, self.num_heads, C // self.num_heads)), (0, 2, 1, 3))

        attn = self.q_matmul_k(q, k)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x_cls = ops.reshape(ops.transpose(self.attn_matmul_v(attn, v), (0, 2, 1, 3)), (B, 1, C))
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


class LayerScaleBlockCA(nn.Cell):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 act_layer: nn.Cell = nn.GELU,
                 norm_layer: nn.Cell = nn.LayerNorm,
                 init_values: float = 1e-4) -> None:
        super(LayerScaleBlockCA, self).__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = ClassAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm2 = norm_layer((dim,))
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       drop=drop_rate,
                       act_layer=act_layer)

        self.gamma_1 = Parameter(init_values * ops.ones((dim), ms.float32), requires_grad=True)
        self.gamma_2 = Parameter(init_values * ops.ones((dim), ms.float32), requires_grad=True)

    def construct(self, x: Tensor, x_cls: Tensor) -> Tensor:
        u = ops.concat((x_cls, x), axis=1)
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))

        return x_cls


class AttentionTalkingHead(nn.Cell):
    """
    Talking head is a trick for multi-head attention,
    which has two more linear map before and after
    the softmax compared to normal attention.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 attn_drop_rate: float = 0.,
                 proj_drop_rate: float = 0.) -> None:
        super(AttentionTalkingHead, self).__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads."
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(1 - attn_drop_rate)

        self.proj = nn.Dense(dim, dim, has_bias=False)

        self.proj_l = nn.Dense(num_heads, num_heads, has_bias=False)
        self.proj_w = nn.Dense(num_heads, num_heads, has_bias=False)

        self.proj_drop = nn.Dropout(1 - proj_drop_rate)

        self.softmax = nn.Softmax(axis=-1)

        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)

    def construct(self, x) -> Tensor:
        B, N, C = x.shape
        qkv = ops.reshape(self.qkv(x), (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = ops.unstack(qkv, axis=0)
        q = ops.mul(q, self.scale)

        attn = self.q_matmul_k(q, k)

        attn = ops.transpose(attn, (0, 2, 3, 1))
        attn = self.proj_l(attn)
        attn = ops.transpose(attn, (0, 3, 1, 2))
        attn = self.softmax(attn)
        attn = ops.transpose(attn, (0, 2, 3, 1))
        attn = self.proj_w(attn)
        attn = ops.transpose(attn, (0, 3, 1, 2))

        attn = self.attn_drop(attn)

        x = self.attn_matmul_v(attn, v)
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LayerScaleBlockSA(nn.Cell):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 act_layer: nn.Cell = nn.GELU,
                 norm_layer: nn.Cell = nn.LayerNorm,
                 init_values: float = 1e-4) -> None:
        super(LayerScaleBlockSA, self).__init__()

        self.norm1 = norm_layer((dim,))
        self.attn = AttentionTalkingHead(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = norm_layer((dim,))
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       drop=drop_rate,
                       act_layer=act_layer)

        self.gamma_1 = Parameter(init_values * ops.ones((dim), ms.float32), requires_grad=True)
        self.gamma_2 = Parameter(init_values * ops.ones((dim), ms.float32), requires_grad=True)

    def construct(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class CaiT(nn.Cell):
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_layer: nn.Cell = nn.LayerNorm,
                 act_layer: nn.Cell = nn.GELU,
                 init_values: float = 1e-4,
                 depth_token_only: int = 2,
                 mlp_ratio_clstk: float = 4.0) -> None:
        super(CaiT, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(image_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=in_channels,
                                      embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        zeros = ops.Zeros()
        self.cls_token = Parameter(zeros((1, 1, embed_dim), ms.float32))
        self.pos_embed = Parameter(zeros((1, num_patches, embed_dim), ms.float32))
        self.pos_drop = nn.Dropout(1 - drop_rate)

        dpr = [drop_path_rate for i in range(depth)]

        self.blocks = []
        self.blocks_token_only = []

        self.blocks = nn.CellList([
            LayerScaleBlockSA(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.blocks_token_only = nn.CellList([
            LayerScaleBlockCA(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                act_layer=act_layer,
                norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth_token_only)])

        self.norm = norm_layer((embed_dim,))

        self.head = nn.Dense(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.pos_embed = init.initializer(TruncatedNormal(sigma=0.02), self.pos_embed.shape, ms.float32)
        self.cls_token = init.initializer(TruncatedNormal(sigma=0.02), self.cls_token.shape, ms.float32)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for name, m in self.cells_and_names():
            if isinstance(m, nn.Dense):
                m.weight = init.initializer(TruncatedNormal(sigma=0.02), m.weight.shape, ms.float32)
                if m.bias is not None:
                    m.bias.set_data(init.initializer(init.Constant(0), m.bias.shape))
            elif isinstance(m, nn.LayerNorm):
                m.beta.set_data(init.initializer(init.Constant(0), m.beta.shape))
                m.gamma.set_data(init.initializer(init.Constant(1), m.gamma.shape))

    def forward_features(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = ops.broadcast_to(self.cls_token, (B, -1, -1))

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i , blk in enumerate(self.blocks):
            x = blk(x)
        for i , blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x, cls_tokens)

        x = ops.concat((cls_tokens, x), axis=1)

        x = self.norm(x)
        return x[:, 0]

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.head(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def cait_xxs24_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> CaiT:
    model = CaiT(img_size=224, patch_size=16, in_channels=in_channels, num_classes=num_classes,
                 embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=False,
                 norm_layer=partial(nn.LayerNorm, epsilon=1e-6), init_values=1e-5, depth_token_only=2,
                 **kwargs)

    if pretrained:
        load_pretrained(model, _cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def cait_xs24_384(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> CaiT:
    model = CaiT(img_size=384, patch_size=16, in_channels=in_channels, num_classes=num_classes,
                 embed_dim=288, depth=24, num_heads=6, mlp_ratio=4, qkv_bias=False,
                 norm_layer=partial(nn.LayerNorm, epsilon=1e-6), init_values=1e-5, depth_token_only=2,
                 **kwargs)

    if pretrained:
        load_pretrained(model, _cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def cait_s24_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> CaiT:
    model = CaiT(img_size=224, patch_size=16, in_channels=in_channels, num_classes=num_classes,
                 embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=False,
                 norm_layer=partial(nn.LayerNorm, epsilon=1e-6), init_values=1e-5, depth_token_only=2,
                 **kwargs)

    if pretrained:
        load_pretrained(model, _cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def cait_s24_384(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> CaiT:
    model = CaiT(img_size=384, patch_size=16, in_channels=in_channels, num_classes=num_classes,
                 embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=False,
                 norm_layer=partial(nn.LayerNorm, epsilon=1e-6), init_values=1e-5, depth_token_only=2,
                 **kwargs)

    if pretrained:
        load_pretrained(model, _cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def cait_s36_384(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> CaiT:
    model = CaiT(img_size=384, patch_size=16, in_channels=in_channels, num_classes=num_classes,
                 embed_dim=384, depth=36, num_heads=8, mlp_ratio=4, qkv_bias=False,
                 norm_layer=partial(nn.LayerNorm, epsilon=1e-6), init_values=1e-6, depth_token_only=2,
                 **kwargs)

    if pretrained:
        load_pretrained(model, _cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def cait_m36_384(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> CaiT:
    model = CaiT(img_size=384, patch_size=16, in_channels=in_channels, num_classes=num_classes,
                 embed_dim=768, depth=36, num_heads=16, mlp_ratio=4, qkv_bias=False,
                 norm_layer=partial(nn.LayerNorm, epsilon=1e-6), init_values=1e-6, depth_token_only=2,
                 **kwargs)

    if pretrained:
        load_pretrained(model, _cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def cait_m48_448(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> CaiT:
    model = CaiT(img_size=448, patch_size=16, in_channels=in_channels, num_classes=num_classes,
                 embed_dim=768, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=False,
                 norm_layer=partial(nn.LayerNorm, epsilon=1e-6), init_values=1e-6, depth_token_only=2,
                 **kwargs)

    if pretrained:
        load_pretrained(model, _cfg, num_classes=num_classes, in_channels=in_channels)
    return model
