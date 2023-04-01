"""
MindSpore implementation of `edgenext`.
Refer to EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications.
"""

import math
from typing import Tuple

import numpy as np

import mindspore as ms
import mindspore.common.initializer as init
from mindspore import Parameter, Tensor, nn, ops

from .layers.drop_path import DropPath
from .layers.identity import Identity
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "EdgeNeXt",
    "edgenext_xx_small",
    "edgenext_x_small",
    "edgenext_small",
    "edgenext_base",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "first_conv": "conv_0.conv",
        "classifier": "last_linear",
        **kwargs,
    }


default_cfgs = {
    "edgenext_xx_small": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/edgenext/edgenext_xx_small-afc971fb.ckpt",
        input_size=(3, 256, 256)),
    "edgenext_x_small": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/edgenext/edgenext_x_small-a200c6fc.ckpt",
        input_size=(3, 256, 256)),
    "edgenext_small": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/edgenext/edgenext_small-f530c372.ckpt",
        input_size=(3, 256, 256)),
    "edgenext_base": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/edgenext/edgenext_base-4335e9dc.ckpt",
        input_size=(3, 256, 256)),
}


def ssplit(x: Tensor, dim, width):
    B, C, H, W = x.shape
    if C % width == 0:
        return ops.split(x, dim, C // width)
    else:
        begin = 0
        temp = []
        while begin + width < C:
            temp.append(x[:, begin : begin + width, :, :])
            begin += width
        temp.append(x[:, begin:, :, :])
        return temp


class LayerNorm(nn.LayerNorm):
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


class PositionalEncodingFourier(nn.Cell):
    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1, has_bias=True)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def construct(self, B, H, W):
        mask = Tensor(np.zeros((B, H, W))).astype(ms.bool_)
        not_mask = ~mask

        y_embed = not_mask.cumsum(1, dtype=ms.float32)
        x_embed = not_mask.cumsum(2, dtype=ms.float32)

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = ms.numpy.arange(self.hidden_dim, dtype=ms.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = ops.stack((ops.sin(pos_x[:, :, :, 0::2]),
                           ops.cos(pos_x[:, :, :, 1::2])), axis=4)
        s1, s2, s3, _, _ = pos_x.shape
        pos_x = ops.reshape(pos_x, (s1, s2, s3, -1))
        pos_y = ops.stack((ops.sin(pos_y[:, :, :, 0::2]),
                           ops.cos(pos_y[:, :, :, 1::2])), axis=4)
        s1, s2, s3, _, _ = pos_y.shape
        pos_y = ops.reshape(pos_y, (s1, s2, s3, -1))
        pos = ops.transpose(ops.concat((pos_y, pos_x), axis=3), (0, 3, 1, 2))
        pos = self.token_projection(pos)
        return pos


class ConvEncoder(nn.Cell):
    def __init__(
        self,
        dim,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        expan_ratio=4,
        kernel_size=7,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, pad_mode="pad", padding=kernel_size // 2, group=dim,
                                has_bias=True)
        self.norm = LayerNorm((dim,), epsilon=1e-6)
        self.pwconv1 = nn.Dense(dim, expan_ratio * dim)
        self.act = nn.GELU(approximate=False)
        self.pwconv2 = nn.Dense(expan_ratio * dim, dim)

        self.gamma1 = (
            Parameter(Tensor(layer_scale_init_value * np.ones(dim), ms.float32), requires_grad=True)
            if layer_scale_init_value > 0.0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()

    def construct(self, x: Tensor) -> Tensor:
        input = x
        x = self.dwconv(x)

        x = ops.transpose(x, (0, 2, 3, 1))
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma1 is not None:
            x = self.gamma1 * x
        x = ops.transpose(x, (0, 3, 1, 2))
        x = input + self.drop_path(x)
        return x


class SDTAEncoder(nn.Cell):
    def __init__(
        self,
        dim,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        expan_ratio=4,
        use_pos_emb=True,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.0,
        drop=0.0,
        scales=1,
    ):
        super().__init__()
        width = max(int(math.ceil(dim / scales)), int(math.floor(dim // scales)))
        self.width = width
        if scales == 1:
            self.nums = 1
        else:
            self.nums = scales - 1
        convs = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, pad_mode="pad", padding=1, group=width, has_bias=True))
        self.convs = nn.CellList(convs)

        self.pos_embd = None
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=dim)
        self.norm_xca = LayerNorm((dim,), epsilon=1e-6)
        self.gamma_xca = Parameter(Tensor(layer_scale_init_value * np.ones(dim), ms.float32),
                                   requires_grad=True) if layer_scale_init_value > 0. else None
        self.xca = XCA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm = LayerNorm((dim,), epsilon=1e-6)
        self.pwconv1 = nn.Dense(dim, expan_ratio * dim)
        self.act = nn.GELU(approximate=False)
        self.pwconv2 = nn.Dense(expan_ratio * dim, dim)
        self.gamma = Parameter(Tensor(layer_scale_init_value * np.ones((dim)), ms.float32),
                               requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

    def construct(self, x: Tensor) -> Tensor:
        input = x

        spx = ssplit(x, 1, self.width)
        sp = None
        out = None
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            if i == 0:
                out = sp
            else:
                out = ops.concat((out, sp), 1)
        x = ops.concat((out, spx[self.nums]), 1)
        # XCA
        B, C, H, W = x.shape
        x = ops.reshape(x, (B, C, H * W))
        x = ops.transpose(x, (0, 2, 1))
        if self.pos_embd is not None:
            pos_encoding = ops.transpose(ops.reshape(self.pos_embd(B, H, W), (B, -1, x.shape[1])), (0, 2, 1))
            x = x + pos_encoding
        x = x + self.drop_path(self.gamma_xca * self.xca(self.norm_xca(x)))
        x = x.astype(ms.float32)
        x = ops.reshape(x, (B, H, W, C))
        # Inverted Bottleneck
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = ops.transpose(x, (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class XCA(nn.Cell):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = Parameter(Tensor(np.ones((num_heads, 1, 1)), ms.float32))

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(1 - attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(1 - proj_drop)

    def construct(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = ops.reshape(self.qkv(x), (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = ops.transpose(q, (0, 1, 3, 2))
        k = ops.transpose(k, (0, 1, 3, 2))
        v = ops.transpose(v, (0, 1, 3, 2))
        l2_normalize = ops.L2Normalize(-1)
        q = l2_normalize(q)
        k = l2_normalize(k)
        attn = (ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))) * self.temperature
        # -------------------
        attn = ops.Softmax(-1)(attn)
        attn = self.attn_drop(attn)
        x = ops.reshape(ops.transpose((ops.matmul(attn, v)), (0, 3, 1, 2)), (B, N, C))
        # # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class EdgeNeXt(nn.Cell):
    r"""EdgeNeXt model class, based on
    `"Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision" <https://arxiv.org/abs/2206.10589>`_

    Args:
        in_channels: number of input channels. Default: 3
        num_classes: number of classification classes. Default: 1000
        depths: the depths of each layer. Default: [0, 0, 0, 3]
        dims: the middle dim of each layer. Default: [24, 48, 88, 168]
        global_block: number of global block. Default: [0, 0, 0, 3]
        global_block_type: type of global block. Default: ['None', 'None', 'None', 'SDTA']
        drop_path_rate: Stochastic Depth. Default: 0.
        layer_scale_init_value: value of layer scale initialization. Default: 1e-6
        head_init_scale: scale of head initialization. Default: 1.
        expan_ratio: ratio of expansion. Default: 4
        kernel_sizes: kernel sizes of different stages. Default: [7, 7, 7, 7]
        heads: number of attention heads. Default: [8, 8, 8, 8]
        use_pos_embd_xca: use position embedding in xca or not. Default: [False, False, False, False]
        use_pos_embd_global: use position embedding globally or not. Default: False
        d2_scales: scales of splitting channels
    """
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[24, 48, 88, 168],
                 global_block=[0, 0, 0, 3], global_block_type=["None", "None", "None", "SDTA"],
                 drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1., expan_ratio=4,
                 kernel_sizes=[7, 7, 7, 7], heads=[8, 8, 8, 8], use_pos_embd_xca=[False, False, False, False],
                 use_pos_embd_global=False, d2_scales=[2, 3, 4, 5], **kwargs):
        super().__init__()
        for g in global_block_type:
            assert g in ["None", "SDTA"]
        if use_pos_embd_global:
            self.pos_embd = PositionalEncodingFourier(dim=dims[0])
        else:
            self.pos_embd = None
        self.downsample_layers = nn.CellList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.SequentialCell(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4, has_bias=True),
            LayerNorm((dims[0],), epsilon=1e-6, norm_axis=1),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.SequentialCell(
                LayerNorm((dims[i],), epsilon=1e-6, norm_axis=1),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2, has_bias=True),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.CellList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = list(np.linspace(0, drop_path_rate, sum(depths)))
        cur = 0
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                if j > depths[i] - global_block[i] - 1:
                    if global_block_type[i] == "SDTA":
                        stage_blocks.append(SDTAEncoder(dim=dims[i], drop_path=dp_rates[cur + j],
                                                        expan_ratio=expan_ratio, scales=d2_scales[i],
                                                        use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i]))
                    else:
                        raise NotImplementedError
                else:
                    stage_blocks.append(ConvEncoder(dim=dims[i], drop_path=dp_rates[cur + j],
                                                    layer_scale_init_value=layer_scale_init_value,
                                                    expan_ratio=expan_ratio, kernel_size=kernel_sizes[i]))

            self.stages.append(nn.SequentialCell(*stage_blocks))
            cur += depths[i]
        self.norm = nn.LayerNorm((dims[-1],), epsilon=1e-6)  # Final norm layer
        self.head = nn.Dense(dims[-1], num_classes)

        # self.head_dropout = nn.Dropout(kwargs["classifier_dropout"])
        self.head_dropout = nn.Dropout(1.0)
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
            elif isinstance(cell, (nn.LayerNorm)):
                cell.gamma.set_data(init.initializer(init.One(), cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(init.initializer(init.Zero(), cell.beta.shape, cell.beta.dtype))
        self.head.weight.set_data(self.head.weight * self.head_init_scale)
        self.head.bias.set_data(self.head.bias * self.head_init_scale)

    def forward_features(self, x):
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        if self.pos_embd is not None:
            B, C, H, W = x.shape
            x = x + self.pos_embd(B, H, W)
        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # Global average pooling, (N, C, H, W) -> (N, C)

    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(self.head_dropout(x))
        return x


@register_model
def edgenext_xx_small(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> EdgeNeXt:
    """Get edgenext_xx_small model.
        Refer to the base class `models.EdgeNeXt` for more details."""
    default_cfg = default_cfgs["edgenext_xx_small"]
    model = EdgeNeXt(
        depths=[2, 2, 6, 2],
        dims=[24, 48, 88, 168],
        expan_ratio=4,
        global_block=[0, 1, 1, 1],
        global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
        use_pos_embd_xca=[False, True, False, False],
        kernel_sizes=[3, 5, 7, 9],
        heads=[4, 4, 4, 4],
        d2_scales=[2, 2, 3, 4],
        **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def edgenext_x_small(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> EdgeNeXt:
    """Get edgenext_x_small model.
    Refer to the base class `models.EdgeNeXt` for more details."""
    default_cfg = default_cfgs["edgenext_x_small"]
    model = EdgeNeXt(
        depths=[3, 3, 9, 3],
        dims=[32, 64, 100, 192],
        expan_ratio=4,
        num_classes=num_classes,
        global_block=[0, 1, 1, 1],
        global_block_type=["None", "SDTA", "SDTA", "SDTA"],
        use_pos_embd_xca=[False, True, False, False],
        kernel_sizes=[3, 5, 7, 9],
        heads=[4, 4, 4, 4],
        d2_scales=[2, 2, 3, 4],
        **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def edgenext_small(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> EdgeNeXt:
    """Get edgenext_small model.
    Refer to the base class `models.EdgeNeXt` for more details."""
    default_cfg = default_cfgs["edgenext_small"]
    model = EdgeNeXt(
        depths=[3, 3, 9, 3],
        dims=[48, 96, 160, 304],
        expan_ratio=4,
        num_classes=num_classes,
        global_block=[0, 1, 1, 1],
        global_block_type=["None", "SDTA", "SDTA", "SDTA"],
        use_pos_embd_xca=[False, True, False, False],
        kernel_sizes=[3, 5, 7, 9],
        d2_scales=[2, 2, 3, 4],
        **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def edgenext_base(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> EdgeNeXt:
    """Get edgenext_base model.
    Refer to the base class `models.EdgeNeXt` for more details."""
    default_cfg = default_cfgs["edgenext_base"]
    model = EdgeNeXt(
        depths=[3, 3, 9, 3],
        dims=[80, 160, 288, 584],
        expan_ratio=4,
        num_classes=num_classes,
        global_block=[0, 1, 1, 1],
        global_block_type=["None", "SDTA", "SDTA", "SDTA"],
        use_pos_embd_xca=[False, True, False, False],
        kernel_sizes=[3, 5, 7, 9],
        d2_scales=[2, 2, 3, 4],
        **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model
