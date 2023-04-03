"""
MindSpore implementation of `crossvit`.
Refer to crossvit: Cross-Attention Multi-Scale Vision Transformer for Image Classification
"""

import numpy as np

import mindspore as ms
import mindspore.common.initializer as init
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.common.initializer import TruncatedNormal

from .layers.drop_path import DropPath
from .layers.helpers import to_2tuple
from .layers.identity import Identity
from .layers.mlp import Mlp
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "crossvit15",
    "crossvit18",
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    "crossvit_15": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/crossvit/crossvit_15-eaa43c02.ckpt"),
    "crossvit_18": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/crossvit/crossvit_18-ca0a2e43.ckpt"),
}


class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(1.0 - attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(1.0 - proj_drop)

    def construct(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        batchmatual = ops.BatchMatMul(transpose_b=True)
        attn = batchmatual(q, k) * self.scale
        softmax = nn.Softmax()
        attn = softmax(attn)
        attn = self.attn_drop(attn)

        batchmatual2 = ops.BatchMatMul()
        x = batchmatual2(attn, v)
        x = ops.transpose(x, (0, 2, 1, 3))
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Cell):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else ops.Identity()
        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def construct(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, multi_conv=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if multi_conv:
            if patch_size[0] == 12:
                self.proj = nn.SequentialCell(
                    nn.Conv2d(in_chans, embed_dim // 4, pad_mode='pad', kernel_size=7, stride=4, padding=3),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, pad_mode='pad', kernel_size=3, stride=3, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 2, embed_dim, pad_mode='pad', kernel_size=3, stride=1, padding=1),
                )
            elif patch_size[0] == 16:
                self.proj = nn.SequentialCell(
                    nn.Conv2d(in_chans, embed_dim // 4, pad_mode='pad', kernel_size=7, stride=4, padding=3),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, pad_mode='pad', kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 2, embed_dim, pad_mode='pad', kernel_size=3, stride=2, padding=1),
                )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, pad_mode='valid',
                                  has_bias=True)

    def construct(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints

        # assert H == self.img_size[0] and W == self.img_size[1], \
        # f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)
        x = ops.transpose(x, (0, 2, 1))
        return x


class CrossAttention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.wk = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.wv = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(1.0 - attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(1.0 - proj_drop)

    def construct(self, x: Tensor) -> Tensor:
        B, N, C = x.shape  # 3,3,16
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads)
        q = ops.transpose(q, (0, 2, 1, 3))  # B1C -> B1H(C/H) -> BH1(C/H) 3 8 1 2

        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads)
        k = ops.transpose(k, (0, 2, 1, 3))  # BNC -> BNH(C/H) -> BHN(C/H) 3832

        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads)
        v = ops.transpose(v, (0, 2, 1, 3))  # BNC -> BNH(C/H) -> BHN(C/H)3832

        batchmatual = ops.BatchMatMul(transpose_b=True)
        attn = batchmatual(q, k) * self.scale
        softmax = nn.Softmax()
        attn = softmax(attn)
        attn = self.attn_drop(attn)
        batchmatual2 = ops.BatchMatMul()
        x = batchmatual2(attn, v)
        x = ops.transpose(x, (0, 2, 1, 3))
        x = x.reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossAttentionBlock(nn.Cell):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()

        self.norm1 = norm_layer((dim,))
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer((dim,))
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer(), drop=drop)

    def construct(self, x: Tensor) -> Tensor:
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class MultiScaleBlock(nn.Cell):
    def __init__(self, dim, patches, depth, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        num_branches = len(dim)
        self.num_branches = num_branches
        blocks = []
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                          drop=drop, attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))
            if len(tmp) != 0:
                blocks.append(nn.SequentialCell(tmp))
        if len(blocks) == 0:
            self.blocks = None
        else:
            self.blocks = nn.CellList(blocks)

        projs = []
        for d in range(num_branches):
            if dim[d] == dim[(d + 1) % num_branches] and False:
                tmp = [Identity()]
            else:
                tmp = [norm_layer((dim[d],), epsilon=1e-6), act_layer(), nn.Dense(dim[d], dim[(d + 1) % num_branches])]
            projs.append(nn.SequentialCell(tmp))
        self.projs = nn.CellList(projs)

        fusion = []
        for d in range(num_branches):
            d_ = (d + 1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:  # backward capability:
                tmp2 = [CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                                            qk_scale=qk_scale,
                                            drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1],
                                            norm_layer=norm_layer,
                                            has_mlp=False)]
                fusion.append(nn.SequentialCell(tmp2))
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                                                   qk_scale=qk_scale,
                                                   drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1],
                                                   norm_layer=norm_layer,
                                                   has_mlp=False))
                fusion.append(nn.SequentialCell(tmp))
        self.fusion = nn.CellList(fusion)
        revert_projs = []
        for d in range(num_branches):
            if dim[(d + 1) % num_branches] == dim[d] and False:
                tmp = [Identity()]
            else:
                tmp = [norm_layer((dim[(d + 1) % num_branches],), epsilon=1e-6), act_layer(),
                       nn.Dense(dim[(d + 1) % num_branches], dim[d])]
            revert_projs.append(nn.SequentialCell(tmp))
        self.revert_projs = nn.CellList(revert_projs)

    def construct(self, x: Tensor) -> Tensor:
        outs_b = []
        i = 0
        for block in self.blocks:
            outs_b.append(block(x[i]))
            i = i + 1
        # only take the cls token out
        proj_cls_token = []
        j = 0
        for proj in self.projs:
            proj_cls_token.append(proj(outs_b[j][:, 0:1]))
            j = j + 1
        outs = []
        for i in range(self.num_branches):
            a = proj_cls_token[i]
            b = outs_b[(i + 1) % self.num_branches][:, 1:, ...]
            con = ops.Concat(1)
            tmp = con((a, b))
            tmp = self.fusion[i](tmp)
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            tmp = con((reverted_proj_cls_token, outs_b[i][:, 1:, ...]))
            outs.append(tmp)
        return outs


def _compute_num_patches(img_size, patches):
    return [i // p * i // p for i, p in zip(img_size, patches)]


def interploate(self, x, output_size, size):
    B, N, C = x.shape
    H, W = size


class VisionTransformer(nn.Cell):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=(224, 224), patch_size=(8, 16), in_channels=3, num_classes=1000, embed_dim=(192, 384),
                 depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]),
                 num_heads=(6, 12), mlp_ratio=(2., 2., 4.), qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, multi_conv=False):
        super().__init__()

        self.num_classes = num_classes
        if not isinstance(img_size, list):
            img_size = to_2tuple(img_size)
        self.img_size = img_size

        num_patches = _compute_num_patches(img_size, patch_size)
        self.num_branches = len(patch_size)

        patch_embed = []
        if hybrid_backbone is None:
            b = []
            for i in range(self.num_branches):
                c = ms.Parameter(Tensor(np.zeros([1, 1 + num_patches[i], embed_dim[i]], np.float32)),
                                 name='pos_embed.' + str(i))
                b.append(c)
            b = tuple(b)
            self.pos_embed = ms.ParameterTuple(b)
            for im_s, p, d in zip(img_size, patch_size, embed_dim):
                patch_embed.append(
                    PatchEmbed(img_size=im_s, patch_size=p, in_chans=in_channels, embed_dim=d, multi_conv=multi_conv))
            self.patch_embed = nn.CellList(patch_embed)

        d = []
        for i in range(self.num_branches):
            c = ms.Parameter(Tensor(np.zeros([1, 1, embed_dim[i]], np.float32)), name='cls_token.' + str(i))
            d.append(c)
        d = tuple(d)
        self.cls_token = ms.ParameterTuple(d)
        self.pos_drop = nn.Dropout(1.0 - drop_rate)

        total_depth = sum([sum(x[-2:]) for x in depth])
        dpr = np.linspace(0, drop_path_rate, total_depth)  # stochastic depth decay rule
        dpr_ptr = 0
        self.blocks = nn.CellList()
        for idx, block_cfg in enumerate(depth):
            curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim, num_patches, block_cfg, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=dpr_,
                                  norm_layer=norm_layer)
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = nn.CellList([norm_layer((embed_dim[i],), epsilon=1e-6) for i in range(self.num_branches)])
        self.head = nn.CellList([nn.Dense(embed_dim[i], num_classes) if num_classes > 0 else Identity() for i in
                                 range(self.num_branches)])

        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                tensor1 = init.initializer(TruncatedNormal(sigma=.02), self.pos_embed[i].data.shape, ms.float32)
                self.pos_embed[i].set_data(tensor1)
            tensor2 = init.initializer(TruncatedNormal(sigma=.02), self.cls_token[i].data.shape, ms.float32)
            self.cls_token[i].set_data(tensor2)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(init.TruncatedNormal(sigma=.02), cell.weight.data.shape))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(init.Constant(0), cell.bias.shape))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(init.initializer(init.Constant(1), cell.gamma.shape))
                cell.beta.set_data(init.initializer(init.Constant(0), cell.beta.shape))

    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Dense(self.embed_dim, num_classes) if num_classes > 0 else Identity()

    def forward_features(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        xs = []
        # print(x)
        for i in range(self.num_branches):
            x_ = ops.interpolate(x, sizes=(self.img_size[i], self.img_size[i]), mode='bilinear') if H != self.img_size[
                i] else x
            tmp = self.patch_embed[i](x_)
            z = self.cls_token[i].shape
            y = Tensor(np.ones((B, z[1], z[2])), dtype=mstype.float32)
            cls_tokens = self.cls_token[i]
            cls_tokens = cls_tokens.expand_as(y)  # stole cls_tokens impl from Phil Wang, thanks
            con = ops.Concat(1)
            cls_tokens = cls_tokens.astype("float32")
            tmp = tmp.astype("float32")
            tmp = con((cls_tokens, tmp))
            tmp = tmp + self.pos_embed[i]
            tmp = self.pos_drop(tmp)
            xs.append(tmp)

        for blk in self.blocks:
            xs = blk(xs)

        # NOTE: was before branch token section, move to here to assure all branch token are before layer norm
        k = 0
        xs2 = []
        for x in xs:
            xs2.append(self.norm[k](x))
            k = k + 1
        xs = xs2
        out = []
        for x in xs:
            out.append(x[:, 0])
        return out

    def forward_head(self, x: Tensor) -> Tensor:
        ce_logits = []
        zz = 0
        for c in x:
            ce_logits.append(self.head[zz](c))
            zz = zz + 1
        z = ops.stack([ce_logits[0], ce_logits[1]])
        op = ops.ReduceMean(keep_dims=False)
        ce_logits = op(z, 0)
        return ce_logits

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def crossvit15(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> VisionTransformer:
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
                              num_heads=[6, 6], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=nn.LayerNorm, in_channels=in_channels, num_classes=num_classes, **kwargs)
    default_cfg = default_cfgs["crossvit_15"]
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def crossvit18(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> VisionTransformer:
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
                              num_heads=[7, 7], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=nn.LayerNorm, in_channels=in_channels, num_classes=num_classes, **kwargs)
    default_cfg = default_cfgs["crossvit_18"]
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model
