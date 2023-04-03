"""
MindSpore implementation of `PVTv2`.
Refer to PVTv2: PVTv2: Improved Baselines with Pyramid Vision Transformer
"""
import math
from functools import partial

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common import initializer as weight_init

from .layers import DropPath, Identity
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "PyramidVisionTransformerV2",
    "pvt_v2_b0",
    "pvt_v2_b1",
    "pvt_v2_b2",
    "pvt_v2_b3",
    "pvt_v2_b4",
    "pvt_v2_b5",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "patch_embed_list.0.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "pvt_v2_b0": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/pvt_v2/pvt_v2_b0-1c4f6683.ckpt"),
    "pvt_v2_b1": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/pvt_v2/pvt_v2_b1-3ceb171a.ckpt"),
    "pvt_v2_b2": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/pvt_v2/pvt_v2_b2-0565d18e.ckpt"),
    "pvt_v2_b3": _cfg(url=""),
    "pvt_v2_b4": _cfg(url=""),
    "pvt_v2_b5": _cfg(url=""),
}


class DWConv(nn.Cell):
    """Depthwise separable convolution"""

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, has_bias=True, group=dim)

    def construct(self, x, H, W):
        B, N, C = x.shape
        x = ops.transpose(x, (0, 2, 1)).view((B, C, H, W))
        x = self.dwconv(x)
        x = ops.transpose(x.view((B, C, H * W)), (0, 2, 1))

        return x


class Mlp(nn.Cell):
    """MLP with depthwise separable convolution"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(1 - drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU()

    def construct(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Cell):
    """Linear Spatial Reduction Attention"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
        super().__init__()
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

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, has_bias=True)
                self.norm = nn.LayerNorm([dim])

        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, has_bias=True)
            self.norm = nn.LayerNorm([dim])
            self.act = nn.GELU()

    def construct(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x)
        q = ops.reshape(q, (B, N, self.num_heads, C // self.num_heads))
        q = ops.transpose(q, (0, 2, 1, 3))

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = ops.reshape(ops.transpose(x, (0, 2, 1)), (B, C, H, W))

                x_ = self.sr(x_)
                x_ = ops.transpose(ops.reshape(x_, (B, C, -1)), (0, 2, 1))
                x_ = self.norm(x_)

                kv = self.kv(x_)
                kv = ops.transpose(ops.reshape(kv, (B, -1, 2, self.num_heads, C // self.num_heads)), (2, 0, 3, 1, 4))
            else:
                kv = self.kv(x)
                kv = ops.transpose(ops.reshape(kv, (B, -1, 2, self.num_heads, C // self.num_heads)), (2, 0, 3, 1, 4))

        else:
            x_ = ops.reshape(ops.transpose(x, (0, 2, 1)), (B, C, H, W))
            x_ = self.sr(self.pool(x_))
            x_ = ops.reshape(ops.transpose(x_, (0, 2, 1)), (B, C, -1))
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = ops.transpose(ops.reshape(self.kv(x_), (B, -1, 2, self.num_heads, C // self.num_heads)),
                               (2, 0, 3, 1, 4))
        k, v = kv[0], kv[1]

        attn = self.qk_batmatmul(q, k) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = self.batmatmul(attn, v)
        x = ops.reshape(ops.transpose(x, (0, 2, 1, 3)), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Cell):
    """Block with Linear Spatial Reduction Attention and Convolutional Feed-Forward"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False, block_id=0):
        super().__init__()
        self.norm1 = norm_layer([dim])

        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()

        self.norm2 = norm_layer([dim])

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

    def construct(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Cell):
    """Overlapping Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, has_bias=True)
        self.norm = nn.LayerNorm([embed_dim])

    def construct(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = ops.transpose(ops.reshape(x, (B, C, H * W)), (0, 2, 1))
        x = self.norm(x)

        return x, H, W


class PyramidVisionTransformerV2(nn.Cell):
    r"""Pyramid Vision Transformer V2 model class, based on
    `"PVTv2: Improved Baselines with Pyramid Vision Transformer" <https://arxiv.org/abs/2106.13797>`_

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
        linear(bool) :  use linear SRA.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        start = Tensor(0, mindspore.float32)
        stop = Tensor(drop_path_rate, mindspore.float32)
        dpr = [float(x) for x in ops.linspace(start, stop, sum(depths))]  # stochastic depth decay rule
        cur = 0

        patch_embed_list = []
        block_list = []
        norm_list = []

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.CellList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear, block_id=j)
                for j in range(depths[i])])

            norm = norm_layer([embed_dims[i]])

            cur += depths[i]

            patch_embed_list.append(patch_embed)
            block_list.append(block)
            norm_list.append(norm)
        self.patch_embed_list = nn.CellList(patch_embed_list)
        self.block_list = nn.CellList(block_list)
        self.norm_list = nn.CellList(norm_list)
        # classification head
        self.head = nn.Dense(embed_dims[3], num_classes) if num_classes > 0 else Identity()
        self._initialize_weights()

    def freeze_patch_emb(self):
        self.patch_embed_list[0].requires_grad = False

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

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = self.patch_embed_list[i]
            block = self.block_list[i]
            norm = self.norm_list[i]
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != self.num_stages - 1:
                x = ops.transpose(ops.reshape(x, (B, H, W, -1)), (0, 3, 1, 2))

        return x.mean(axis=1)

    def forward_head(self, x: Tensor) -> Tensor:
        return self.head(x)

    def construct(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)

        return x


@register_model
def pvt_v2_b0(
    pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs
) -> PyramidVisionTransformerV2:
    """Get PVTV2-b0 model
    Refer to the base class "models.PVTv2" for more details.
    """
    default_cfg = default_cfgs["pvt_v2_b0"]
    model = PyramidVisionTransformerV2(
        in_chans=in_channels, num_classes=num_classes,
        patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def pvt_v2_b1(
    pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs
) -> PyramidVisionTransformerV2:
    """Get PVTV2-b1 model
    Refer to the base class "models.PVTv2" for more details.
    """
    default_cfg = default_cfgs["pvt_v2_b1"]
    model = PyramidVisionTransformerV2(
        in_chans=in_channels, num_classes=num_classes,
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def pvt_v2_b2(
    pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs
) -> PyramidVisionTransformerV2:
    """Get PVTV2-b2 model
    Refer to the base class "models.PVTv2" for more details.
    """
    default_cfg = default_cfgs["pvt_v2_b2"]
    model = PyramidVisionTransformerV2(
        in_chans=in_channels, num_classes=num_classes,
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def pvt_v2_b3(
    pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs
) -> PyramidVisionTransformerV2:
    """Get PVTV2-b3 model
    Refer to the base class "models.PVTv2" for more details.
    """
    default_cfg = default_cfgs["pvt_v2_b3"]
    model = PyramidVisionTransformerV2(
        in_chans=in_channels, num_classes=num_classes,
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def pvt_v2_b4(
    pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs
) -> PyramidVisionTransformerV2:
    """Get PVTV2-b4 model
    Refer to the base class "models.PVTv2" for more details.
    """
    default_cfg = default_cfgs["pvt_v2_b4"]
    model = PyramidVisionTransformerV2(
        in_chans=in_channels, num_classes=num_classes,
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def pvt_v2_b5(
    pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs
) -> PyramidVisionTransformerV2:
    """Get PVTV2-b5 model
    Refer to the base class "models.PVTv2" for more details.
    """
    default_cfg = default_cfgs["pvt_v2_b5"]
    model = PyramidVisionTransformerV2(
        in_chans=in_channels, num_classes=num_classes,
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
