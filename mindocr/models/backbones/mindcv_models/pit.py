"""
MindSpore implementation of `PiT`.
Refer to Rethinking Spatial Dimensions of Vision Transformers.
"""

import math
from typing import List

import numpy as np

import mindspore.common.initializer as init
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import nn, ops

from .layers import DropPath, Identity
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "pit_ti",
    "pit_xs",
    "pit_s",
    "pit_b",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "fixed_input_size": True,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "pit_ti_224": _cfg(url=""),
    "pit_xs_224": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/pit/pit_xs-fea0d37e.ckpt"),
    "pit_s_224": _cfg(url=""),
    "pit_b_224": _cfg(url=""),
}


class conv_embedding(nn.Cell):
    """define embedding layer using conv2d"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=patch_size,
            stride=stride,
            pad_mode="pad",
            padding=padding,
            has_bias=True,
        )

    def construct(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return x


class conv_head_pooling(nn.Cell):
    """define pooling layer using conv in spatial tokens with an additional fully-connected layer
    (to adjust the channel size to match the spatial tokens)"""

    def __init__(
        self,
        in_feature: int,
        out_feature: int,
        stride: int,
        pad_mode: str = "pad",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_feature,
            out_feature,
            kernel_size=stride + 1,
            padding=stride // 2,
            stride=stride,
            pad_mode=pad_mode,
            group=in_feature,
            has_bias=True,
        )
        self.fc = nn.Dense(in_channels=in_feature, out_channels=out_feature, has_bias=True)

    def construct(self, x, cls_token):
        x = self.conv(x)
        cls_token = self.fc(cls_token)

        return x, cls_token


class Attention(nn.Cell):
    """define multi-head self attention block"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        # get pair-wise relative position index for each token inside the window
        self.q = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.k = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.v = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(keep_prob=1 - attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(keep_prob=1 - proj_drop)
        self.softmax = nn.Softmax(axis=-1)

        self.batchmatmul = ops.BatchMatMul()

    def construct(self, x):
        B, N, C = x.shape
        q = ops.reshape(self.q(x), (B, N, self.num_heads, C // self.num_heads)) * self.scale
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.reshape(self.k(x), (B, N, self.num_heads, C // self.num_heads))
        k = ops.transpose(k, (0, 2, 3, 1))
        v = ops.reshape(self.v(x), (B, N, self.num_heads, C // self.num_heads))
        v = ops.transpose(v, (0, 2, 1, 3))

        attn = self.batchmatmul(q, k)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = self.batchmatmul(attn, v)
        x = ops.reshape(ops.transpose(x, (0, 2, 1, 3)), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Cell):
    """define the basic block of PiT"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.cell = nn.GELU,
        norm_layer: nn.cell = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer((dim,), epsilon=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer((dim,), epsilon=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Mlp(nn.Cell):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: nn.cell = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=True)
        self.act = act_layer()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features, has_bias=True)
        self.drop = nn.Dropout(keep_prob=1.0 - drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Transformer(nn.Cell):
    """define the transformer block of PiT"""

    def __init__(
        self,
        base_dim: List[int],
        depth: List[int],
        heads: List[int],
        mlp_ratio: float,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_prob: float = None,
    ) -> None:
        super().__init__()
        self.layers = nn.CellList([])
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        self.blocks = nn.CellList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_prob[i],
                    norm_layer=nn.LayerNorm,
                )
                for i in range(depth)
            ]
        )

    def construct(self, x, cls_tokens):
        h, w = x.shape[2:4]
        x = ops.reshape(x, (x.shape[0], x.shape[1], h * w))
        x = ops.transpose(x, (0, 2, 1))
        token_length = cls_tokens.shape[1]
        x = ops.concat((cls_tokens, x), axis=1)
        for blk in self.blocks:
            x = blk(x)

        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        x = ops.transpose(x, (0, 2, 1))
        x = ops.reshape(x, (x.shape[0], x.shape[1], h, w))
        return x, cls_tokens


class PoolingTransformer(nn.Cell):
    r"""PiT model class, based on
    `"Rethinking Spatial Dimensions of Vision Transformers"
    <https://arxiv.org/abs/2103.16302>`
    Args:
        image_size (int) : images input size.
        patch_size (int) : image patch size.
        stride (int) : stride of the depthwise conv.
        base_dims (List[int]) : middle dim of each layer.
        depth (List[int]) : model block depth of each layer.
        heads (List[int]) : number of heads of multi-head attention of each layer
        mlp_ratio (float) : ratio of hidden features in Mlp.
        num_classes (int) : number of classification classes. Default: 1000.
        in_chans (int) : number the channels of the input. Default: 3.
        attn_drop_rate (float) : attention layers dropout rate. Default: 0.
        drop_rate (float) : dropout rate. Default: 0.
        drop_path_rate (float) : drop path rate. Default: 0.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        stride: int,
        base_dims: List[int],
        depth: List[int],
        heads: List[int],
        mlp_ratio: float,
        num_classes: int = 1000,
        in_chans: int = 3,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        total_block = sum(depth)
        padding = 0
        block_idx = 0

        width = math.floor((image_size + 2 * padding - patch_size) / stride + 1)

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes

        self.patch_size = patch_size
        self.pos_embed = Parameter(Tensor(np.random.randn(1, base_dims[0] * heads[0], width, width), mstype.float32))
        self.patch_embed = conv_embedding(in_chans, base_dims[0] * heads[0], patch_size, stride, padding)
        self.cls_token = Parameter(Tensor(np.random.randn(1, 1, base_dims[0] * heads[0]), mstype.float32))

        self.pos_drop = nn.Dropout(keep_prob=1.0 - drop_rate)
        self.tile = ops.Tile()

        self.transformers = nn.CellList([])
        self.pools = nn.CellList([])

        for stage in range(len(depth)):
            drop_path_prob = [drop_path_rate * i / total_block for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]
            self.transformers.append(
                Transformer(
                    base_dims[stage], depth[stage], heads[stage], mlp_ratio, drop_rate, attn_drop_rate, drop_path_prob
                )
            )
            if stage < len(heads) - 1:
                self.pools.append(
                    conv_head_pooling(
                        base_dims[stage] * heads[stage], base_dims[stage + 1] * heads[stage + 1], stride=2
                    )
                )

        self.norm = nn.LayerNorm((base_dims[-1] * heads[-1],), epsilon=1e-6)

        self.embed_dim = base_dims[-1] * heads[-1]

        # Classifier head
        if num_classes > 0:
            self.head = nn.Dense(in_channels=base_dims[-1] * heads[-1], out_channels=num_classes, has_bias=True)
        else:
            self.head = Identity()

        self.pos_embed.set_data(
            init.initializer(init.TruncatedNormal(sigma=0.02), self.pos_embed.shape, self.pos_embed.dtype)
        )
        self.cls_token.set_data(
            init.initializer(init.TruncatedNormal(sigma=0.02), self.cls_token.shape, self.cls_token.dtype)
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """init_weights"""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(init.initializer(init.One(), cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(init.initializer(init.Zero(), cell.beta.shape, cell.beta.dtype))
            if isinstance(cell, nn.Conv2d):
                n = cell.kernel_size[0] * cell.kernel_size[1] * cell.in_channels
                cell.weight.set_data(
                    init.initializer(init.Uniform(math.sqrt(1.0 / n)), cell.weight.shape, cell.weight.dtype)
                )
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer(init.Uniform(math.sqrt(1.0 / n)), cell.bias.shape, cell.bias.dtype)
                    )
            if isinstance(cell, nn.Dense):
                init_range = 1.0 / np.sqrt(cell.weight.shape[0])
                cell.weight.set_data(init.initializer(init.Uniform(init_range), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(init.Uniform(init_range), cell.bias.shape, cell.bias.dtype))

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)

        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)

        cls_tokens = self.tile(self.cls_token, (x.shape[0], 1, 1))

        for stage in range(len(self.pools)):
            x, cls_tokens = self.transformers[stage](x, cls_tokens)
            x, cls_tokens = self.pools[stage](x, cls_tokens)
        x, cls_tokens = self.transformers[-1](x, cls_tokens)

        cls_tokens = self.norm(cls_tokens)

        return cls_tokens

    def forward_head(self, x: Tensor) -> Tensor:
        cls_token = self.head(x[:, 0])
        return cls_token

    def construct(self, x: Tensor) -> Tensor:
        cls_token = self.forward_features(x)
        cls_token = self.forward_head(cls_token)
        return cls_token


@register_model
def pit_ti(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> PoolingTransformer:
    """Get PiT-Ti model.
    Refer to the base class `models.PoolingTransformer` for more details."""
    default_cfg = default_cfgs["pit_ti_224"]
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4.0,
        num_classes=num_classes,
        in_chans=in_channels,
        **kwargs
    )

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def pit_b(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> PoolingTransformer:
    """Get PiT-B model.
    Refer to the base class `models.PoolingTransformer` for more details."""
    default_cfg = default_cfgs["pit_b_224"]
    model = PoolingTransformer(
        image_size=224,
        patch_size=14,
        stride=7,
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4.0,
        num_classes=num_classes,
        in_chans=in_channels,
        **kwargs
    )

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def pit_s(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> PoolingTransformer:
    """Get PiT-S model.
    Refer to the base class `models.PoolingTransformer` for more details."""
    default_cfg = default_cfgs["pit_s_224"]
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4.0,
        num_classes=num_classes,
        in_chans=in_channels,
        **kwargs
    )

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def pit_xs(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> PoolingTransformer:
    """Get PiT-XS model.
    Refer to the base class `models.PoolingTransformer` for more details."""
    default_cfg = default_cfgs["pit_xs_224"]
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4.0,
        num_classes=num_classes,
        in_chans=in_channels,
        **kwargs
    )

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
