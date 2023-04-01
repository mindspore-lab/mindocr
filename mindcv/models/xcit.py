"""
MindSpore implementation of XCiT
Refer to: XCiT: Cross-Covariance Image Transformers
"""
from functools import partial

import numpy as np

import mindspore
import mindspore.common.initializer as weight_init
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import nn, numpy, ops

from .layers import DropPath
from .layers.mlp import Mlp
from .registry import register_model
from .utils import _ntuple, load_pretrained

__all__ = [
    'XCiT',
    'xcit_tiny_12_p16',
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': '',
        'classifier': '',
        **kwargs
    }


default_cfgs = {
    'xcit_tiny_12_p16': _cfg(
        url='https://download.mindspore.cn/toolkits/mindcv/xcit/xcit_tiny_12_p16_224-1b1c9301.ckpt'),
}

to_2tuple = _ntuple(2)


class PositionalEncodingFourier(nn.Cell):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self,
                 hidden_dim: int = 32,
                 dim: int = 768,
                 temperature=10000
                 ) -> None:
        super().__init__()
        self.token_projection = nn.Conv2d(
            hidden_dim * 2, dim, kernel_size=1, has_bias=True)
        self.scale = 2 * np.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def construct(self, B, H, W) -> Tensor:
        mask = Tensor(np.zeros((B, H, W)).astype(bool))
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=mstype.float32)
        x_embed = not_mask.cumsum(2, dtype=mstype.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = numpy.arange(self.hidden_dim, dtype=mstype.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = ops.stack((ops.sin(pos_x[:, :, :, 0::2]),
                           ops.cos(pos_x[:, :, :, 1::2])), 4)
        x1, x2, x3, x4, x5 = pos_x.shape
        pos_x = ops.reshape(pos_x, (x1, x2, x3, x4 * x5))
        pos_y = ops.stack((ops.sin(pos_y[:, :, :, 0::2]),
                           ops.cos(pos_y[:, :, :, 1::2])), 4)
        y1, y2, y3, y4, y5 = pos_y.shape
        pos_y = ops.reshape(pos_y, (y1, y2, y3, y4 * y5))
        pos = ops.transpose(ops.concat((pos_y, pos_x), 3), (0, 3, 1, 2))
        pos = self.token_projection(pos)
        return pos


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.SequentialCell([
        nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, pad_mode='pad', has_bias=False
        ),
        nn.BatchNorm2d(out_planes, use_batch_statistics=True)
    ])


class ConvPatchEmbed(nn.Cell):
    """ Image to Patch Embedding using multiple convolutional layers
    """

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 768
                 ) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        if patch_size[0] == 16:
            self.proj = nn.SequentialCell([
                conv3x3(3, embed_dim // 8, 2),
                nn.GELU(),
                conv3x3(embed_dim // 8, embed_dim // 4, 2),
                nn.GELU(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            ])
        elif patch_size[0] == 8:
            self.proj = nn.SequentialCell([
                conv3x3(3, embed_dim // 4, 2),
                nn.GELU(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            ])
        else:
            raise ValueError(
                "For convolutional projection, patch size has to be in [8, 16]")

    def construct(self, x, padding_size=None) -> Tensor:
        x = self.proj(x)
        B, C, Hp, Wp = x.shape
        x = ops.reshape(x, (B, C, Hp * Wp))
        x = x.transpose(0, 2, 1)

        return x, (Hp, Wp)


class LPI(nn.Cell):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows
    to augment the implicit communcation performed by the block diagonal scatter attention.
    Implemented using 2 layers of separable 3x3 convolutions with GeLU and BatchNorm2d
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 drop=0., kernel_size=3) -> None:
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                               padding=padding, pad_mode='pad', group=out_features, has_bias=True)
        self.act = act_layer()
        self.bn = nn.BatchNorm2d(in_features, use_batch_statistics=True)
        self.conv2 = nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                               padding=padding, pad_mode='pad', group=out_features, has_bias=True)

    def construct(self, x, H, W) -> Tensor:
        B, N, C = x.shape
        x = ops.reshape(ops.transpose(x, (0, 2, 1)), (B, C, H, W))
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = ops.transpose(ops.reshape(x, (B, C, N)), (0, 2, 1))

        return x


class ClassAttention(nn.Cell):
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Dense(
            in_channels=dim, out_channels=dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(keep_prob=1 - attn_drop)
        self.proj = nn.Dense(in_channels=dim, out_channels=dim)
        self.proj_drop = nn.Dropout(keep_prob=1 - proj_drop)
        self.softmax = nn.Softmax(axis=-1)

        self.attn_matmul_v = ops.BatchMatMul()

    def construct(self, x: Tensor) -> Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = ops.unstack(qkv, axis=0)
        qc = q[:, :, 0:1]
        attn_cls = (qc * k).sum(-1) * self.scale
        attn_cls = self.softmax(attn_cls)
        attn_cls = self.attn_drop(attn_cls)

        attn_cls = ops.expand_dims(attn_cls, 2)
        cls_tkn = self.attn_matmul_v(attn_cls, v)
        cls_tkn = ops.transpose(cls_tkn, (0, 2, 1, 3))
        cls_tkn = ops.reshape(cls_tkn, (B, 1, C))
        cls_tkn = self.proj(cls_tkn)
        x = ops.concat((self.proj_drop(cls_tkn), x[:, 1:]), axis=1)
        return x


class ClassAttentionBlock(nn.Cell):
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, eta=None,
                 tokens_norm=False):
        super().__init__()
        self.norm1 = norm_layer([dim])
        self.attn = ClassAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop
        )

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else ops.Identity()
        self.norm2 = norm_layer([dim])
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop)

        # LayerScale Initialization (no layerscale when None)
        if eta is not None:
            self.gamma1 = Parameter(
                eta * ops.Ones()((dim), mstype.float32), requires_grad=True)
            self.gamma2 = Parameter(
                eta * ops.Ones()((dim), mstype.float32), requires_grad=True)
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

        # FIXME: A hack for models pre-trained with layernorm over all the tokens not just the CLS
        self.tokens_norm = tokens_norm

    def construct(self, x, H, W, mask=None):

        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))

        if self.tokens_norm:
            x = self.norm2(x)
        else:
            x[:, 0:1] = self.norm2(x[:, 0:1])
        x_res = x
        cls_token = x[:, 0:1]
        cls_token = self.gamma2 * self.mlp(cls_token)
        x = ops.concat((cls_token, x[:, 1:]), axis=1)
        x = x_res + self.drop_path(x)
        return x


class XCA(nn.Cell):

    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = Parameter(
            ops.Ones()((num_heads, 1, 1), mstype.float32))
        self.qkv = nn.Dense(
            in_channels=dim, out_channels=dim * 3, has_bias=qkv_bias)
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax(axis=-1)
        self.attn_drop = nn.Dropout(keep_prob=1.0 - attn_drop)
        self.attn_matmul_v = ops.BatchMatMul()
        self.proj = nn.Dense(in_channels=dim, out_channels=dim)
        self.proj_drop = nn.Dropout(keep_prob=1.0 - proj_drop)

    def construct(self, x):
        B, N, C = x.shape

        qkv = ops.reshape(
            self.qkv(x), (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = ops.unstack(qkv, axis=0)

        q = ops.transpose(q, (0, 1, 3, 2))
        k = ops.transpose(k, (0, 1, 3, 2))
        v = ops.transpose(v, (0, 1, 3, 2))

        attn = self.q_matmul_k(q, k) * self.temperature
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = self.attn_matmul_v(attn, v)
        x = ops.transpose(x, (0, 3, 1, 2))
        x = ops.reshape(x, (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class XCABlock(nn.Cell):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 num_tokens=196, eta=None):
        super().__init__()
        self.norm1 = norm_layer([dim])
        self.attn = XCA(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer([dim])

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop)

        self.norm3 = norm_layer([dim])
        self.local_mp = LPI(in_features=dim, act_layer=act_layer)

        self.gamma1 = Parameter(
            eta * ops.ones(dim, mstype.float32), requires_grad=True)
        self.gamma2 = Parameter(
            eta * ops.ones(dim, mstype.float32), requires_grad=True)
        self.gamma3 = Parameter(
            eta * ops.ones(dim, mstype.float32), requires_grad=True)

    def construct(self, x, H, W):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma3 *
                               self.local_mp(self.norm3(x), H, W))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class XCiT(nn.Cell):
    r"""XCiT model class, based on
    `"XCiT: Cross-Covariance Image Transformers" <https://arxiv.org/abs/2106.09681>`_
    Args:
        img_size (int, tuple): input image size
        patch_size (int, tuple): patch size
        in_chans (int): number of input channels
        num_classes (int): number of classes for classification head
        embed_dim (int): embedding dimension
        depth (int): depth of transformer
        num_heads (int): number of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): enable bias for qkv if True
        qk_scale (float): override default qk scale of head_dim ** -0.5 if set
        drop_rate (float): dropout rate
        attn_drop_rate (float): attention dropout rate
        drop_path_rate (float): stochastic depth rate
        norm_layer: (nn.Module): normalization layer
        cls_attn_layers: (int) Depth of Class attention layers
        use_pos: (bool) whether to use positional encoding
        eta: (float) layerscale initialization value
        tokens_norm: (bool) Whether to normalize all tokens or just the cls_token in the CA
    """

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: int = 4.,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_layer: nn.Cell = None,
                 cls_attn_layers: int = 2,
                 use_pos: bool = True,
                 patch_proj: str = 'linear',
                 eta: float = None,
                 tokens_norm: bool = False):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, epsilon=1e-6)

        self.patch_embed = ConvPatchEmbed(img_size=img_size, embed_dim=embed_dim,
                                          patch_size=patch_size)

        num_patches = self.patch_embed.num_patches

        self.cls_token = Parameter(
            ops.zeros((1, 1, embed_dim), mstype.float32))
        self.pos_drop = nn.Dropout(keep_prob=1.0 - drop_rate)

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.CellList([
            XCABlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, num_tokens=num_patches, eta=eta)
            for i in range(depth)])

        self.cls_attn_blocks = nn.CellList([
            ClassAttentionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer,
                eta=eta, tokens_norm=tokens_norm)
            for i in range(cls_attn_layers)])
        self.norm = norm_layer([embed_dim])
        self.head = nn.Dense(
            in_channels=embed_dim, out_channels=num_classes) if num_classes > 0 else ops.Identity()

        self.pos_embeder = PositionalEncodingFourier(dim=embed_dim)
        self.use_pos = use_pos

        # Classifier head
        self.cls_token.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                        self.cls_token.shape,
                                                        self.cls_token.dtype))
        self._init_weights()

    def _init_weights(self) -> None:
        for name, m in self.cells_and_names():
            if isinstance(m, nn.Dense):
                m.weight = weight_init.initializer(weight_init.TruncatedNormal(
                    sigma=0.02), m.weight.shape, mindspore.float32)
                if m.bias is not None:
                    m.bias.set_data(weight_init.initializer(
                        weight_init.Constant(0), m.bias.shape))
            elif isinstance(m, nn.LayerNorm):
                m.beta.set_data(weight_init.initializer(
                    weight_init.Constant(0), m.beta.shape))
                m.gamma.set_data(weight_init.initializer(
                    weight_init.Constant(1), m.gamma.shape))

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        if self.use_pos:
            pos_encoding = self.pos_embeder(B, Hp, Wp).reshape(
                B, -1, x.shape[1]).transpose(0, 2, 1)
            x = x + pos_encoding
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x, Hp, Wp)
        cls_tokens = ops.broadcast_to(self.cls_token, (B, -1, -1))
        cls_tokens = ops.cast(cls_tokens, x.dtype)
        x = ops.concat((cls_tokens, x), 1)

        for blk in self.cls_attn_blocks:
            x = blk(x, Hp, Wp)
        return self.norm(x)[:, 0]

    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def xcit_tiny_12_p16(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> XCiT:
    """Get xcit_tiny_12_p16 model.
    Refer to the base class 'models.XCiT' for more details.
    """
    default_cfg = default_cfgs['xcit_tiny_12_p16']
    model = XCiT(
        patch_size=16, num_classes=num_classes, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), eta=1.0, tokens_norm=True, **kwargs)
    if pretrained:
        load_pretrained(model, default_cfg,
                        num_classes=num_classes, in_channels=in_channels)

    return model
