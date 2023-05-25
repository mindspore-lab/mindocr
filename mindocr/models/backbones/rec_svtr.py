from typing import Any, Callable, List, Optional, Tuple, Union

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Parameter, Tensor

from ._registry import register_backbone, register_backbone_class
from .mindcv_models.layers import DropPath

__all__ = ["SVTRNet", "rec_svtr"]


class ConvBNLayer(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias_attr: bool = False,
        groups: int = 1,
        act: Callable[..., nn.Cell] = nn.GELU,
    ) -> None:
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode="pad",
            padding=padding,
            group=groups,
            has_bias=bias_attr,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def construct(self, inputs: Tensor) -> Tensor:
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class Identity(nn.Cell):
    def __init__(self) -> None:
        super(Identity, self).__init__()

    def construct(self, input: Tensor) -> Tensor:
        return input


class Mlp(nn.Cell):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: Callable[..., nn.Cell] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(keep_prob=1 - drop)

    def construct(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMixer(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        HW: Tuple[int, int] = [8, 25],
        local_k: Tuple[int, int] = [3, 3],
    ) -> None:
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2d(
            dim,
            dim,
            local_k,
            1,
            pad_mode="pad",
            padding=[
                local_k[0] // 2,
                local_k[0] // 2,
                local_k[1] // 2,
                local_k[1] // 2,
            ],
            group=num_heads,
            has_bias=True,
        )

    def construct(self, x: Tensor) -> Tensor:
        h = self.HW[0]
        w = self.HW[1]
        x = x.transpose([0, 2, 1]).reshape([-1, self.dim, h, w])
        x = self.local_mixer(x)
        x = ops.reshape(x, (x.shape[0], x.shape[1], -1))
        x = x.transpose([0, 2, 1])
        return x


class Attention(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mixer: str = "Global",
        HW: Optional[Tuple[int, int]] = None,
        local_k: Tuple[int, int] = [7, 11],
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(keep_prob=1 - attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(keep_prob=1 - proj_drop)
        self.HW = HW
        if HW is not None:
            H = HW[0]
            W = HW[1]
            self.N = H * W
            self.C = dim
        if mixer == "Local" and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = ops.ones((H * W, H + hk - 1, W + wk - 1), ms.float32)
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h : h + hk, w : w + wk] = 0.0
            mask = mask[:, hk // 2 : H + hk // 2, wk // 2 : W + wk // 2]
            mask = ops.reshape(mask, (mask.shape[0], -1))
            mask_inf = ms.numpy.full([H * W, H * W], float("-inf"), dtype="float32")
            mask = ms.numpy.where(mask < 1, mask, mask_inf)
            self.mask = mask[None, None, ...]
        self.mixer = mixer
        self.matmul = ops.BatchMatMul()

    def construct(self, x: Tensor) -> Tensor:
        if self.HW is not None:
            N = self.N
            C = self.C
        else:
            _, N, C = x.shape
        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (-1, N, 3, self.num_heads, C // self.num_heads))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = self.matmul(q, k.transpose((0, 1, 3, 2)))
        if self.mixer == "Local":
            attn += self.mask
        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = self.matmul(attn, v)
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (-1, N, C))

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mixer: str = "Global",
        local_mixer: Tuple[int, int] = [7, 11],
        HW: Optional[Tuple[int, int]] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Cell] = nn.GELU,
        norm_layer: Union[str, Callable[..., nn.Cell]] = "nn.LayerNorm",
        epsilon: float = 1e-6,
        prenorm: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)([dim], epsilon=epsilon)
        else:
            self.norm1 = norm_layer([dim])
        if mixer == "Global" or mixer == "Local":
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif mixer == "Conv":
            self.mixer = ConvMixer(dim, num_heads=num_heads, HW=HW, local_k=local_mixer)
        else:
            raise TypeError("The mixer must be one of [Global, Local, Conv]")

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)([dim], epsilon=epsilon)
        else:
            self.norm2 = norm_layer([dim])
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.prenorm = prenorm

    def construct(self, x: Tensor) -> Tensor:
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Cell):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size: Tuple[int, int] = [32, 100],
        in_channels: int = 3,
        embed_dim: int = 768,
        sub_num: int = 2,
        patch_size: Tuple[int, int] = [4, 4],
        mode: str = "pope",
    ) -> None:
        super().__init__()
        num_patches = (img_size[1] // (2**sub_num)) * (img_size[0] // (2**sub_num))
        self.img_size = img_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.norm = None
        if mode == "pope":
            if sub_num == 2:
                self.proj = nn.SequentialCell(
                    ConvBNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=False,
                    ),
                    ConvBNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=False,
                    ),
                )
            if sub_num == 3:
                self.proj = nn.SequentialCell(
                    ConvBNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 4,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=False,
                    ),
                    ConvBNLayer(
                        in_channels=embed_dim // 4,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=False,
                    ),
                    ConvBNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=False,
                    ),
                )
        elif mode == "linear":
            self.proj = nn.Conv2d(
                1,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                has_bias=True,
            )
            self.num_patches = (
                img_size[0] // patch_size[0] * img_size[1] // patch_size[1]
            )

    def construct(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = ops.reshape(x, (x.shape[0], x.shape[1], -1))
        x = x.transpose((0, 2, 1))
        return x


class SubSample(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        types: str = "Pool",
        stride: Tuple[int, int] = (2, 1),
        sub_norm: Union[str, Callable[..., nn.Cell]] = "nn.LayerNorm",
        act: Optional[Callable[..., nn.Cell]] = None,
    ) -> None:
        super().__init__()
        self.types = types
        if types == "Pool":
            self.avgpool = nn.AvgPool2d(
                kernel_size=[3, 5], stride=stride, pad_mode="same"
            )
            self.maxpool = nn.MaxPool2d(
                kernel_size=[3, 5], stride=stride, pad_mode="same"
            )
            self.proj = nn.Dense(in_channels, out_channels)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                pad_mode="pad",
                has_bias=False,
            )
        self.norm = eval(sub_norm)([out_channels])
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def construct(self, x: Tensor) -> Tensor:
        if self.types == "Pool":
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = (x1 + x2) * 0.5
            out = self.proj(
                x.reshape((x.shape[0], x.shape[1], -1)).transpose((0, 2, 1))
            )
        else:
            x = self.conv(x)
            x = ops.reshape(x, (x.shape[0], x.shape[1], -1))
            out = x.transpose((0, 2, 1))
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)

        return out


@register_backbone_class
class SVTRNet(nn.Cell):
    def __init__(
        self,
        img_size: Tuple[int, int] = [32, 100],
        in_channels: int = 3,
        embed_dim: List[int] = [64, 128, 256],
        depth: List[int] = [3, 6, 3],
        num_heads: List[int] = [2, 4, 8],
        mixer: List[str] = ["Local"] * 6 + ["Global"] * 6,  # Local, Global, Conv
        local_mixer: List[Tuple[int, int]] = [[7, 11], [7, 11], [7, 11]],
        patch_merging: str = "Conv",  # Conv, Pool, None
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        last_drop: float = 0.1,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: Union[str, Callable[..., nn.Cell]] = "nn.LayerNorm",
        sub_norm: Union[str, Callable[..., nn.Cell]] = "nn.LayerNorm",
        epsilon: float = 1e-6,
        out_channels: int = 192,
        block_unit: str = "Block",
        act: str = "nn.GELU",
        last_stage: bool = True,
        sub_num: int = 2,
        prenorm: int = True,
        use_lenhead: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.prenorm = prenorm
        patch_merging = (
            None
            if patch_merging != "Conv" and patch_merging != "Pool"
            else patch_merging
        )
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim[0],
            sub_num=sub_num,
        )
        num_patches = self.patch_embed.num_patches
        self.HW = [img_size[0] // (2**sub_num), img_size[1] // (2**sub_num)]

        self.pos_embed = Parameter(
            ops.zeros((1, num_patches, embed_dim[0]), ms.float32)
        )

        self.pos_drop = nn.Dropout(keep_prob=1 - drop_rate)
        Block_unit = eval(block_unit)
        dpr = np.linspace(0, drop_path_rate, num=sum(depth))
        self.blocks1 = nn.CellList(
            [
                Block_unit(
                    dim=embed_dim[0],
                    num_heads=num_heads[0],
                    mixer=mixer[0 : depth[0]][i],
                    HW=self.HW,
                    local_mixer=local_mixer[0],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=eval(act),
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[0 : depth[0]][i],
                    norm_layer=norm_layer,
                    epsilon=epsilon,
                    prenorm=prenorm,
                )
                for i in range(depth[0])
            ]
        )
        if patch_merging is not None:
            self.sub_sample1 = SubSample(
                embed_dim[0],
                embed_dim[1],
                sub_norm=sub_norm,
                stride=(2, 1),
                types=patch_merging,
            )
            HW = [self.HW[0] // 2, self.HW[1]]
        else:
            HW = self.HW
        self.patch_merging = patch_merging
        self.blocks2 = nn.CellList(
            [
                Block_unit(
                    dim=embed_dim[1],
                    num_heads=num_heads[1],
                    mixer=mixer[depth[0] : depth[0] + depth[1]][i],
                    HW=HW,
                    local_mixer=local_mixer[1],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=eval(act),
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[depth[0] : depth[0] + depth[1]][i],
                    norm_layer=norm_layer,
                    epsilon=epsilon,
                    prenorm=prenorm,
                )
                for i in range(depth[1])
            ]
        )
        if patch_merging is not None:
            self.sub_sample2 = SubSample(
                embed_dim[1],
                embed_dim[2],
                sub_norm=sub_norm,
                stride=(2, 1),
                types=patch_merging,
            )
            HW = [self.HW[0] // 4, self.HW[1]]
        else:
            HW = self.HW
        self.blocks3 = nn.CellList(
            [
                Block_unit(
                    dim=embed_dim[2],
                    num_heads=num_heads[2],
                    mixer=mixer[depth[0] + depth[1] :][i],
                    HW=HW,
                    local_mixer=local_mixer[2],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=eval(act),
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[depth[0] + depth[1] :][i],
                    norm_layer=norm_layer,
                    epsilon=epsilon,
                    prenorm=prenorm,
                )
                for i in range(depth[2])
            ]
        )
        self.last_stage = last_stage
        if last_stage:
            self.last_conv = nn.Conv2d(
                in_channels=embed_dim[2],
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                pad_mode="pad",
                has_bias=False,
            )
            self.hardswish = nn.HSwish()
            self.dropout = nn.Dropout(keep_prob=1 - last_drop)
        if not prenorm:
            self.norm = eval(norm_layer)([embed_dim[-1]], epsilon=epsilon)
        self.use_lenhead = use_lenhead
        if use_lenhead:
            self.len_conv = nn.Dense(embed_dim[2], self.out_channels)
            self.hardswish_len = nn.HSwish()
            self.dropout_len = nn.Dropout(keep_prob=1 - last_drop)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample1(
                x.transpose([0, 2, 1]).reshape(
                    [-1, self.embed_dim[0], self.HW[0], self.HW[1]]
                )
            )
        for blk in self.blocks2:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample2(
                x.transpose([0, 2, 1]).reshape(
                    [-1, self.embed_dim[1], self.HW[0] // 2, self.HW[1]]
                )
            )
        for blk in self.blocks3:
            x = blk(x)
        if not self.prenorm:
            x = self.norm(x)
        return x

    def construct(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = self.forward_features(x)

        if self.use_lenhead:
            len_x = self.len_conv(x.mean(1))
            len_x = self.dropout_len(self.hardswish_len(len_x))
        else:
            len_x = -1

        if self.last_stage:
            if self.patch_merging is not None:
                h = self.HW[0] // 4
            else:
                h = self.HW[0]
            x = ops.mean(
                x.transpose([0, 2, 1]).reshape([-1, self.embed_dim[2], h, self.HW[1]]),
                axis=2,
                keep_dims=True,
            )
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)

        if self.use_lenhead:
            return [x], [len_x]
        else:
            return [x]


@register_backbone
def rec_svtr(pretrained: bool = False, **kwargs):
    model = SVTRNet(**kwargs)

    # load pretrained weights
    if pretrained is True:
        raise NotImplementedError(
            "The default pretrained checkpoint for `rec_svtr` backbone does not exist."
        )

    return model
