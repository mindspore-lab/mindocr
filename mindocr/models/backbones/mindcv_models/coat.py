"""
CoaT architecture.
Modified from timm/models/vision_transformer.py
"""
from typing import Union

import numpy as np

import mindspore
import mindspore.common.initializer as init
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, ms_function
from mindspore.numpy import split

from .layers.drop_path import DropPath
from .layers.identity import Identity
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "coat_tiny",
    "coat_mini",
    "coat_small",
    "coat_lite_tiny",
    "coat_lite_mini",
    "coat_lite_small",
    "coat_lite_medium",
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': .9,
        'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'coat_tiny': _cfg(url=''),
    'coat_mini': _cfg(url=''),
    'coat_small': _cfg(url=''),
    'coat_lite_tiny': _cfg(url='https://download.mindspore.cn/toolkits/mindcv/coat/coat_lite_tiny-fa7bf894.ckpt'),
    'coat_lite_mini': _cfg(url='https://download.mindspore.cn/toolkits/mindcv/coat/coat_lite_mini-55a52f05.ckpt'),
    'coat_lite_small': _cfg(url=''),
    'coat_lite_medium': _cfg(url='')
}


class Mlp(nn.Cell):
    """MLP Cell"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=True)
        self.act = nn.GELU(approximate=False)
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features, has_bias=True)
        self.drop = nn.Dropout(keep_prob=1.0 - drop)

    def construct(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvRelPosEnc(nn.Cell):

    def __init__(
        self,
        Ch,
        h,
        window
    ) -> None:
        super().__init__()

        if isinstance(window, int):
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.CellList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2
            cur_conv = nn.Conv2d(in_channels=cur_head_split * Ch,
                                 out_channels=cur_head_split * Ch,
                                 kernel_size=(cur_window, cur_window),
                                 padding=(padding_size, padding_size, padding_size, padding_size),
                                 dilation=(dilation, dilation),
                                 group=cur_head_split * Ch,
                                 pad_mode='pad',
                                 has_bias=True)
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]
        self.idx1 = self.channel_splits[0]
        self.idx2 = self.channel_splits[0] + self.channel_splits[1]

    @ms_function
    def construct(self, q, v, size) -> Tensor:

        B, h, N, Ch = q.shape
        H, W = size

        q_img = q[:, :, 1:, :]
        v_img = v[:, :, 1:, :]

        v_img = ops.transpose(v_img, (0, 1, 3, 2))
        v_img = ops.reshape(v_img, (B, h * Ch, H, W))

        v_img_list = split(x=v_img, indices_or_sections=[self.idx1, self.idx2], axis=1)

        conv_v_img_list = []
        i = 0
        for conv in self.conv_list:
            conv_v_img_list.append(conv(v_img_list[i]))
            i = i + 1
        conv_v_img = ops.concat(conv_v_img_list, axis=1)
        conv_v_img = ops.reshape(conv_v_img, (B, h, Ch, H * W))
        conv_v_img = ops.transpose(conv_v_img, (0, 1, 3, 2))

        EV_hat_img = q_img * conv_v_img
        zero = ops.Zeros()((B, h, 1, Ch), q.dtype)
        EV_hat = ops.concat((zero, EV_hat_img), axis=2)
        return EV_hat


class FactorAtt_ConvRelPosEnc(nn.Cell):
    """Factorized attention with convolutional relative position encoding class."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        shared_crpe=None
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.k = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.v = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(keep_prob=1 - attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(keep_prob=1 - proj_drop)
        self.softmax = nn.Softmax(axis=-1)
        self.batch_matmul = ops.BatchMatMul()

        self.crpe = shared_crpe

    def construct(self, x, size) -> Tensor:
        B, N, C = x.shape
        q = ops.reshape(self.q(x), (B, N, self.num_heads, C // self.num_heads))
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.reshape(self.k(x), (B, N, self.num_heads, C // self.num_heads))
        k = ops.transpose(k, (0, 2, 3, 1))
        v = ops.reshape(self.v(x), (B, N, self.num_heads, C // self.num_heads))
        v = ops.transpose(v, (0, 2, 1, 3))

        k_softmax = self.softmax(k)
        factor_att = self.batch_matmul(q, k_softmax)
        factor_att = self.batch_matmul(factor_att, v)

        crpe = self.crpe(q, v, size=size)

        x = ops.mul(self.scale, factor_att)
        x = ops.add(x, crpe)
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (B, N, C))

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ConvPosEnc(nn.Cell):
    """ Convolutional Position Encoding.
        Note: This module is similar to the conditional position encoding in CPVT.
    """

    def __init__(
        self,
        dim,
        k=3
    ) -> None:
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(in_channels=dim,
                              out_channels=dim,
                              kernel_size=k,
                              stride=1,
                              padding=k // 2,
                              group=dim,
                              pad_mode='pad',
                              has_bias=True)

    def construct(self, x, size) -> Tensor:
        B, N, C = x.shape
        H, W = size

        cls_token, img_tokens = x[:, :1], x[:, 1:]

        feat = ops.transpose(img_tokens, (0, 2, 1))
        feat = ops.reshape(feat, (B, C, H, W))
        x = ops.add(self.proj(feat), feat)

        x = ops.reshape(x, (B, C, H * W))
        x = ops.transpose(x, (0, 2, 1))

        x = ops.concat((cls_token, x), axis=1)
        return x


class SerialBlock(nn.Cell):
    """
    Serial block class.
        Note: In this implementation, each serial block only contains a conv-attention and a FFN (MLP) module.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        shared_cpe=None,
        shared_crpe=None
    ) -> None:
        super().__init__()

        self.cpe = shared_cpe

        self.norm1 = nn.LayerNorm((dim,), epsilon=1e-6)
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(dim,
                                                      num_heads=num_heads,
                                                      qkv_bias=qkv_bias,
                                                      attn_drop=attn_drop,
                                                      proj_drop=drop,
                                                      shared_crpe=shared_crpe
                                                      )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

        self.norm2 = nn.LayerNorm((dim,), epsilon=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def construct(self, x, size) -> Tensor:
        x = x + self.drop_path(self.factoratt_crpe(self.norm1(self.cpe(x, size)), size))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ParallelBlock(nn.Cell):
    """ Parallel block class. """

    def __init__(
        self,
        dims,
        num_heads,
        mlp_ratios=[],
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        shared_cpes=None,
        shared_crpes=None
    ) -> None:
        super().__init__()

        self.cpes = shared_cpes

        self.norm12 = nn.LayerNorm((dims[1],), epsilon=1e-6)
        self.norm13 = nn.LayerNorm((dims[2],), epsilon=1e-6)
        self.norm14 = nn.LayerNorm((dims[3],), epsilon=1e-6)
        self.factoratt_crpe2 = FactorAtt_ConvRelPosEnc(dims[1],
                                                       num_heads=num_heads,
                                                       qkv_bias=qkv_bias,
                                                       attn_drop=attn_drop,
                                                       proj_drop=drop,
                                                       shared_crpe=shared_crpes[1]
                                                       )
        self.factoratt_crpe3 = FactorAtt_ConvRelPosEnc(dims[2],
                                                       num_heads=num_heads,
                                                       qkv_bias=qkv_bias,
                                                       attn_drop=attn_drop,
                                                       proj_drop=drop,
                                                       shared_crpe=shared_crpes[2]
                                                       )
        self.factoratt_crpe4 = FactorAtt_ConvRelPosEnc(dims[3],
                                                       num_heads=num_heads,
                                                       qkv_bias=qkv_bias,
                                                       attn_drop=attn_drop,
                                                       proj_drop=drop,
                                                       shared_crpe=shared_crpes[3]
                                                       )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

        self.norm22 = nn.LayerNorm((dims[1],), epsilon=1e-6)
        self.norm23 = nn.LayerNorm((dims[2],), epsilon=1e-6)
        self.norm24 = nn.LayerNorm((dims[3],), epsilon=1e-6)

        mlp_hidden_dim = int(dims[1] * mlp_ratios[1])
        self.mlp2 = self.mlp3 = self.mlp4 = Mlp(in_features=dims[1], hidden_features=mlp_hidden_dim, drop=drop)

    def upsample(self, x, output_size, size) -> Tensor:
        """ Feature map up-sampling. """
        return self.interpolate(x, output_size=output_size, size=size)

    def downsample(self, x, output_size, size) -> Tensor:
        """ Feature map down-sampling. """
        return self.interpolate(x, output_size=output_size, size=size)

    def interpolate(self, x, output_size, size) -> Tensor:
        """ Feature map interpolation. """
        B, N, C = x.shape
        H, W = size

        cls_token = x[:, :1, :]
        img_tokens = x[:, 1:, :]

        img_tokens = ops.transpose(img_tokens, (0, 2, 1))
        img_tokens = ops.reshape(img_tokens, (B, C, H, W))
        img_tokens = ops.interpolate(img_tokens,
                                     sizes=output_size,
                                     mode='bilinear'
                                     )
        img_tokens = ops.reshape(img_tokens, (B, C, -1))
        img_tokens = ops.transpose(img_tokens, (0, 2, 1))

        out = ops.concat((cls_token, img_tokens), axis=1)
        return out

    def construct(self, x1, x2, x3, x4, sizes) -> tuple:
        _, (H2, W2), (H3, W3), (H4, W4) = sizes

        # Conv-Attention.
        x2 = self.cpes[1](x2, size=(H2, W2))  # Note: x1 is ignored.
        x3 = self.cpes[2](x3, size=(H3, W3))
        x4 = self.cpes[3](x4, size=(H4, W4))

        cur2 = self.norm12(x2)
        cur3 = self.norm13(x3)
        cur4 = self.norm14(x4)
        cur2 = self.factoratt_crpe2(cur2, size=(H2, W2))
        cur3 = self.factoratt_crpe3(cur3, size=(H3, W3))
        cur4 = self.factoratt_crpe4(cur4, size=(H4, W4))
        upsample3_2 = self.upsample(cur3, output_size=(H2, W2), size=(H3, W3))
        upsample4_3 = self.upsample(cur4, output_size=(H3, W3), size=(H4, W4))
        upsample4_2 = self.upsample(cur4, output_size=(H2, W2), size=(H4, W4))
        downsample2_3 = self.downsample(cur2, output_size=(H3, W3), size=(H2, W2))
        downsample3_4 = self.downsample(cur3, output_size=(H4, W4), size=(H3, W3))
        downsample2_4 = self.downsample(cur2, output_size=(H4, W4), size=(H2, W2))
        cur2 = cur2 + upsample3_2 + upsample4_2
        cur3 = cur3 + upsample4_3 + downsample2_3
        cur4 = cur4 + downsample3_4 + downsample2_4
        x2 = x2 + self.drop_path(cur2)
        x3 = x3 + self.drop_path(cur3)
        x4 = x4 + self.drop_path(cur4)

        cur2 = self.norm22(x2)
        cur3 = self.norm23(x3)
        cur4 = self.norm24(x4)
        cur2 = self.mlp2(cur2)
        cur3 = self.mlp3(cur3)
        cur4 = self.mlp4(cur4)
        x2 = x2 + self.drop_path(cur2)
        x3 = x3 + self.drop_path(cur3)
        x4 = x4 + self.drop_path(cur4)

        return x1, x2, x3, x4


class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding """

    def __init__(
        self,
        image_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96
    ) -> None:
        super().__init__()
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [image_size[0] // patch_size[0], image_size[1] // patch_size[1]]

        self.image_size = image_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels=in_chans,
                              out_channels=embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size,
                              pad_mode='valid',
                              has_bias=True)

        self.norm = nn.LayerNorm((embed_dim,), epsilon=1e-5)

    def construct(self, x: Tensor) -> Tensor:
        B = x.shape[0]

        x = ops.reshape(self.proj(x), (B, self.embed_dim, -1))
        x = ops.transpose(x, (0, 2, 1))
        x = self.norm(x)

        return x


class CoaT(nn.Cell):
    """ CoaT class. """

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dims=[0, 0, 0, 0],
        serial_depths=[0, 0, 0, 0],
        parallel_depth=0,
        num_heads=0,
        mlp_ratios=[0, 0, 0, 0],
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        return_interm_layers=False,
        out_features=None,
        crpe_window={3: 2, 5: 3, 7: 3},
        **kwargs
    ) -> None:
        super().__init__()
        self.return_interm_layers = return_interm_layers
        self.out_features = out_features
        self.num_classes = num_classes

        self.patch_embed1 = PatchEmbed(image_size=image_size, patch_size=patch_size,
                                       in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(image_size=image_size // (2**2), patch_size=2,
                                       in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(image_size=image_size // (2**3), patch_size=2,
                                       in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(image_size=image_size // (2**4), patch_size=2,
                                       in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.cls_token1 = mindspore.Parameter(ops.Zeros()((1, 1, embed_dims[0]), mindspore.float32))
        self.cls_token2 = mindspore.Parameter(ops.Zeros()((1, 1, embed_dims[1]), mindspore.float32))
        self.cls_token3 = mindspore.Parameter(ops.Zeros()((1, 1, embed_dims[2]), mindspore.float32))
        self.cls_token4 = mindspore.Parameter(ops.Zeros()((1, 1, embed_dims[3]), mindspore.float32))

        self.cpe1 = ConvPosEnc(dim=embed_dims[0], k=3)
        self.cpe2 = ConvPosEnc(dim=embed_dims[1], k=3)
        self.cpe3 = ConvPosEnc(dim=embed_dims[2], k=3)
        self.cpe4 = ConvPosEnc(dim=embed_dims[3], k=3)

        self.crpe1 = ConvRelPosEnc(Ch=embed_dims[0] // num_heads, h=num_heads, window=crpe_window)
        self.crpe2 = ConvRelPosEnc(Ch=embed_dims[1] // num_heads, h=num_heads, window=crpe_window)
        self.crpe3 = ConvRelPosEnc(Ch=embed_dims[2] // num_heads, h=num_heads, window=crpe_window)
        self.crpe4 = ConvRelPosEnc(Ch=embed_dims[3] // num_heads, h=num_heads, window=crpe_window)

        dpr = drop_path_rate

        self.serial_blocks1 = nn.CellList([
            SerialBlock(
                dim=embed_dims[0], num_heads=num_heads, mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr,
                shared_cpe=self.cpe1, shared_crpe=self.crpe1
            )
            for _ in range(serial_depths[0])]
        )

        self.serial_blocks2 = nn.CellList([
            SerialBlock(
                dim=embed_dims[1], num_heads=num_heads, mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr,
                shared_cpe=self.cpe2, shared_crpe=self.crpe2
            )
            for _ in range(serial_depths[1])]
        )

        self.serial_blocks3 = nn.CellList([
            SerialBlock(
                dim=embed_dims[2], num_heads=num_heads, mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr,
                shared_cpe=self.cpe3, shared_crpe=self.crpe3
            )
            for _ in range(serial_depths[2])]
        )

        self.serial_blocks4 = nn.CellList([
            SerialBlock(
                dim=embed_dims[3], num_heads=num_heads, mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr,
                shared_cpe=self.cpe4, shared_crpe=self.crpe4
            )
            for _ in range(serial_depths[3])]
        )

        self.parallel_depth = parallel_depth
        if self.parallel_depth > 0:
            self.parallel_blocks = nn.CellList([
                ParallelBlock(dims=embed_dims,
                              num_heads=num_heads,
                              mlp_ratios=mlp_ratios,
                              qkv_bias=qkv_bias,
                              drop=drop_rate,
                              attn_drop=attn_drop_rate,
                              drop_path=dpr,
                              shared_cpes=[self.cpe1, self.cpe2, self.cpe3, self.cpe4],
                              shared_crpes=[self.crpe1, self.crpe2, self.crpe3, self.crpe4]
                              )
                for _ in range(parallel_depth)]
            )
        else:
            self.parallel_blocks = None

        if not self.return_interm_layers:
            if self.parallel_blocks is not None:
                self.norm2 = nn.LayerNorm((embed_dims[1],), epsilon=1e-6)
                self.norm3 = nn.LayerNorm((embed_dims[2],), epsilon=1e-6)
            else:
                self.norm2 = None
                self.norm3 = None

            self.norm4 = nn.LayerNorm((embed_dims[3],), epsilon=1e-6)

            if self.parallel_depth > 0:
                self.aggregate = nn.Conv1d(in_channels=3,
                                           out_channels=1,
                                           kernel_size=1,
                                           has_bias=True)
                self.head = nn.Dense(embed_dims[3], num_classes) if num_classes > 0 else Identity()
            else:
                self.aggregate = None
                self.head = nn.Dense(embed_dims[3], num_classes) if num_classes > 0 else Identity()

        self.cls_token1.set_data(init.initializer(init.TruncatedNormal(sigma=.02), self.cls_token1.data.shape))
        self.cls_token2.set_data(init.initializer(init.TruncatedNormal(sigma=.02), self.cls_token2.data.shape))
        self.cls_token3.set_data(init.initializer(init.TruncatedNormal(sigma=.02), self.cls_token3.data.shape))
        self.cls_token4.set_data(init.initializer(init.TruncatedNormal(sigma=.02), self.cls_token4.data.shape))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(init.TruncatedNormal(sigma=.02), cell.weight.data.shape))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(init.Constant(0), cell.bias.shape))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(init.initializer(init.Constant(1.0), cell.gamma.shape))
                cell.beta.set_data(init.initializer(init.Constant(0), cell.beta.shape))

    def insert_cls(self, x, cls_token) -> Tensor:
        t0 = x.shape[0]
        t1 = cls_token.shape[1]
        t2 = cls_token.shape[2]
        y = Tensor(np.ones((t0, t1, t2)))
        cls_tokens = cls_token.expand_as(y)

        x = ops.concat((cls_tokens, x), axis=1)
        return x

    def remove_cls(self, x: Tensor) -> Tensor:
        return x[:, 1:, :]

    def forward_features(self, x0: Tensor) -> Union[dict, Tensor]:
        B = x0.shape[0]

        x1 = self.patch_embed1(x0)
        H1, W1 = self.patch_embed1.patches_resolution
        x1 = self.insert_cls(x1, self.cls_token1)
        for blk in self.serial_blocks1:
            x1 = blk(x1, size=(H1, W1))
        x1_nocls = self.remove_cls(x1)
        x1_nocls = ops.reshape(x1_nocls, (B, H1, W1, -1))
        x1_nocls = ops.transpose(x1_nocls, (0, 3, 1, 2))

        x2 = self.patch_embed2(x1_nocls)
        H2, W2 = self.patch_embed2.patches_resolution
        x2 = self.insert_cls(x2, self.cls_token2)
        for blk in self.serial_blocks2:
            x2 = blk(x2, size=(H2, W2))
        x2_nocls = self.remove_cls(x2)
        x2_nocls = ops.reshape(x2_nocls, (B, H2, W2, -1))
        x2_nocls = ops.transpose(x2_nocls, (0, 3, 1, 2))

        x3 = self.patch_embed3(x2_nocls)
        H3, W3 = self.patch_embed3.patches_resolution
        x3 = self.insert_cls(x3, self.cls_token3)
        for blk in self.serial_blocks3:
            x3 = blk(x3, size=(H3, W3))
        x3_nocls = self.remove_cls(x3)
        x3_nocls = ops.reshape(x3_nocls, (B, H3, W3, -1))
        x3_nocls = ops.transpose(x3_nocls, (0, 3, 1, 2))

        x4 = self.patch_embed4(x3_nocls)
        H4, W4 = self.patch_embed4.patches_resolution
        x4 = self.insert_cls(x4, self.cls_token4)
        for blk in self.serial_blocks4:
            x4 = blk(x4, size=(H4, W4))
        x4_nocls = self.remove_cls(x4)
        x4_nocls = ops.reshape(x4_nocls, (B, H4, W4, -1))
        x4_nocls = ops.transpose(x4_nocls, (0, 3, 1, 2))

        if self.parallel_depth == 0:
            if self.return_interm_layers:
                feat_out = {}
                if 'x1_nocls' in self.out_features:
                    feat_out['x1_nocls'] = x1_nocls
                if 'x2_nocls' in self.out_features:
                    feat_out['x2_nocls'] = x2_nocls
                if 'x3_nocls' in self.out_features:
                    feat_out['x3_nocls'] = x3_nocls
                if 'x4_nocls' in self.out_features:
                    feat_out['x4_nocls'] = x4_nocls
                return feat_out
            else:
                x4 = self.norm4(x4)
                x4_cls = x4[:, 0]
                return x4_cls

        for blk in self.parallel_blocks:
            x1, x2, x3, x4 = blk(x1, x2, x3, x4, sizes=[(H1, W1), (H2, W2), (H3, W3), (H4, W4)])

        if self.return_interm_layers:
            feat_out = {}
            if 'x1_nocls' in self.out_features:
                x1_nocls = x1[:, 1:, :].reshape((B, H1, W1, -1)).transpose((0, 3, 1, 2))
                feat_out['x1_nocls'] = x1_nocls
            if 'x2_nocls' in self.out_features:
                x2_nocls = x2[:, 1:, :].reshape((B, H2, W2, -1)).transpose((0, 3, 1, 2))
                feat_out['x2_nocls'] = x2_nocls
            if 'x3_nocls' in self.out_features:
                x3_nocls = x3[:, 1:, :].reshape((B, H3, W3, -1)).transpose((0, 3, 1, 2))
                feat_out['x3_nocls'] = x3_nocls
            if 'x4_nocls' in self.out_features:
                x4_nocls = x4[:, 1:, :].reshape((B, H4, W4, -1)).transpose((0, 3, 1, 2))
                feat_out['x4_nocls'] = x4_nocls
            return feat_out
        else:
            x2 = self.norm2(x2)
            x3 = self.norm3(x3)
            x4 = self.norm4(x4)
            x2_cls = x2[:, :1]
            x3_cls = x3[:, :1]
            x4_cls = x4[:, :1]
            merged_cls = ops.concat((x2_cls, x3_cls, x4_cls), axis=1)
            merged_cls = self.aggregate(merged_cls).squeeze(axis=1)
            return merged_cls

    def construct(self, x: Tensor) -> Union[dict, Tensor]:
        if self.return_interm_layers:
            return self.forward_features(x)
        else:
            x = self.forward_features(x)
            x = self.head(x)
            return x


@register_model
def coat_tiny(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['coat_mini']
    model = CoaT(in_channels=in_channels, num_classes=num_classes,
                 patch_size=4, embed_dims=[152, 152, 152, 152],
                 serial_depths=[2, 2, 2, 2], parallel_depth=6,
                 num_heads=8, mlp_ratios=[4, 4, 4, 4], **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def coat_mini(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['coat_mini']
    model = CoaT(in_channels=in_channels, num_classes=num_classes,
                 patch_size=4, embed_dims=[152, 216, 216, 216],
                 serial_depths=[2, 2, 2, 2], parallel_depth=6,
                 num_heads=8, mlp_ratios=[4, 4, 4, 4], **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def coat_small(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['coat_small']
    model = CoaT(in_channels=in_channels, num_classes=num_classes,
                 patch_size=4, embed_dims=[152, 320, 320, 320],
                 serial_depths=[2, 2, 2, 2], parallel_depth=6,
                 num_heads=8, mlp_ratios=[4, 4, 4, 4], **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def coat_lite_tiny(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['coat_lite_tiny']
    model = CoaT(in_channels=in_channels, num_classes=num_classes,
                 patch_size=4, embed_dims=[64, 128, 256, 320],
                 serial_depths=[2, 2, 2, 2], parallel_depth=0,
                 num_heads=8, mlp_ratios=[8, 8, 4, 4], **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def coat_lite_mini(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['coat_lite_mini']
    model = CoaT(in_channels=in_channels, num_classes=num_classes,
                 patch_size=4, embed_dims=[64, 128, 320, 512],
                 serial_depths=[2, 2, 2, 2], parallel_depth=0,
                 num_heads=8, mlp_ratios=[8, 8, 4, 4], **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def coat_lite_small(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['coat_lite_small']
    model = CoaT(in_channels=in_channels, num_classes=num_classes,
                 patch_size=4, embed_dims=[64, 128, 320, 512],
                 serial_depths=[3, 4, 6, 3], parallel_depth=0,
                 num_heads=8, mlp_ratios=[8, 8, 4, 4], **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def coat_lite_medium(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['coat_lite_medium']
    model = CoaT(in_channels=in_channels, num_classes=num_classes,
                 patch_size=4, embed_dims=[128, 256, 320, 512],
                 serial_depths=[3, 6, 10, 8], parallel_depth=0,
                 num_heads=8, mlp_ratios=[4, 4, 4, 4], **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model
