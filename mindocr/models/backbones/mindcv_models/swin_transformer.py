"""Define SwinTransformer model"""
from typing import List, Optional, Tuple

import numpy as np

import mindspore.common.initializer as init
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import nn, numpy, ops

from .layers import DropPath, Identity
from .registry import register_model
from .utils import _ntuple, load_pretrained

__all__ = [
    "SwinTransformer",
    "swin_tiny",
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
    "swin_tiny": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/swin/swin_tiny-0ff2f96d.ckpt"),
}

to_2tuple = _ntuple(2)


class Mlp(nn.Cell):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[nn.Cell] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=True)
        self.act = act_layer()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features, has_bias=True)
        self.drop = nn.Dropout(keep_prob=1.0 - drop)

    def construct(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: numpy(num_windows*B, window_size, window_size, C)
    """
    b, h, w, c = x.shape
    x = np.reshape(x, (b, h // window_size, window_size, w // window_size, window_size, c))
    windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, c)
    return windows


class WindowPartition(nn.Cell):
    def __init__(
        self,
        window_size: int,
    ) -> None:
        super(WindowPartition, self).__init__()

        self.window_size = window_size

    def construct(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (b, h, w, c)
            window_size (int): window size

        Returns:
            windows: Tensor(num_windows*b, window_size, window_size, c)
        """
        b, h, w, c = x.shape
        x = ops.reshape(x, (b, h // self.window_size, self.window_size, w // self.window_size, self.window_size, c))
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        x = ops.reshape(x, (b * h * w // (self.window_size**2), self.window_size, self.window_size, c))

        return x


class WindowReverse(nn.Cell):
    def construct(
        self,
        windows: Tensor,
        window_size: int,
        h: int,
        w: int,
    ) -> Tensor:
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            h (int): Height of image
            w (int): Width of image

        Returns:
            x: (B, H, W, C)
        """
        b = windows.shape[0] // (h * w // window_size // window_size)
        x = ops.reshape(windows, (b, h // window_size, w // window_size, window_size, window_size, -1))
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        x = ops.reshape(x, (b, h, w, -1))
        return x


class RelativeBias(nn.Cell):
    def __init__(
        self,
        window_size: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        # define a parameter table of relative position bias
        coords_h = np.arange(self.window_size[0]).reshape(self.window_size[0], 1).repeat(self.window_size[0],
                                                                                         1).reshape(1, -1)
        coords_w = np.arange(self.window_size[1]).reshape(1, self.window_size[1]).repeat(self.window_size[1],
                                                                                         0).reshape(1, -1)
        coords_flatten = np.concatenate([coords_h, coords_w], axis=0)  # 2, Wh, Ww
        relative_coords = coords_flatten[:, :, np.newaxis] - coords_flatten[:, np.newaxis, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.transpose(1, 2, 0)  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.relative_position_index = Tensor(relative_coords.sum(-1).reshape(-1))  # Wh*Ww, Wh*Ww
        self.relative_position_bias_table = Parameter(
            Tensor(np.random.randn((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads),
                   dtype=mstype.float32))  # 2*Wh-1 * 2*Ww-1, nH
        self.one_hot = nn.OneHot(axis=-1, depth=(2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                                 dtype=mstype.float32)
        self.index = Parameter(self.one_hot(self.relative_position_index), requires_grad=False)

    def construct(self) -> Tensor:
        out = ops.matmul(self.index, self.relative_position_bias_table)
        out = ops.reshape(out, (self.window_size[0] * self.window_size[1],
                                self.window_size[0] * self.window_size[1], -1))
        out = ops.transpose(out, (2, 0, 1))
        out = ops.expand_dims(out, 0)
        return out


class WindowAttention(nn.Cell):
    r"""Window based multi-head self attention (W-MSA) Cell with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qZk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if isinstance(dim, tuple) and len(dim) == 1:
            dim = dim[0]
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = Tensor(qk_scale or head_dim**-0.5, mstype.float32)
        self.relative_bias = RelativeBias(self.window_size, num_heads)

        # get pair-wise relative position index for each token inside the window
        self.q = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.k = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.v = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)

        self.attn_drop = nn.Dropout(keep_prob=1.0 - attn_drop)
        self.proj = nn.Dense(in_channels=dim, out_channels=dim, has_bias=True)
        self.proj_drop = nn.Dropout(keep_prob=1.0 - proj_drop)
        self.softmax = nn.Softmax(axis=-1)
        self.batch_matmul = ops.BatchMatMul()

    def construct(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape
        q = ops.reshape(self.q(x), (b_, n, self.num_heads, c // self.num_heads)) * self.scale
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.reshape(self.k(x), (b_, n, self.num_heads, c // self.num_heads))
        k = ops.transpose(k, (0, 2, 3, 1))
        v = ops.reshape(self.v(x), (b_, n, self.num_heads, c // self.num_heads))
        v = ops.transpose(v, (0, 2, 1, 3))

        attn = self.batch_matmul(q, k)
        attn = attn + self.relative_bias()

        if mask is not None:
            nw = mask.shape[1]
            attn = ops.reshape(attn, (b_ // nw, nw, self.num_heads, n, n,)) + mask
            attn = ops.reshape(attn, (-1, self.num_heads, n, n,))
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = ops.reshape(ops.transpose(self.batch_matmul(attn, v), (0, 2, 1, 3)), (b_, n, c))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"


class SwinTransformerBlock(nn.Cell):
    """Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Cell, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Cell, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Optional[nn.Cell] = nn.GELU,
        norm_layer: Optional[nn.Cell] = nn.LayerNorm,
    ) -> None:
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        if isinstance(dim, int):
            dim = (dim,)

        self.norm1 = norm_layer(dim, epsilon=1e-5)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim, epsilon=1e-5)
        mlp_hidden_dim = int((dim[0] if isinstance(dim, tuple) else dim) * mlp_ratio)
        self.mlp = Mlp(in_features=dim[0] if isinstance(dim, tuple) else dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            h_, w_ = self.input_resolution
            img_mask = np.zeros((1, h_, w_, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            # img_mask: [1, 56, 56, 1] window_size: 7
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)
            attn_mask = mask_windows[:, np.newaxis] - mask_windows[:, :, np.newaxis]
            # [64, 49, 49] ==> [1, 64, 1, 49, 49]
            attn_mask = np.expand_dims(attn_mask, axis=1)
            attn_mask = np.expand_dims(attn_mask, axis=0)
            attn_mask = Tensor(np.where(attn_mask == 0, 0.0, -100.0), dtype=mstype.float32)
            self.attn_mask = Parameter(attn_mask, requires_grad=False)
            self.roll_pos = Roll(self.shift_size)
            self.roll_neg = Roll(-self.shift_size)
        else:
            self.attn_mask = None

        self.window_partition = WindowPartition(self.window_size)
        self.window_reverse = WindowReverse()

    def construct(self, x: Tensor) -> Tensor:
        h, w = self.input_resolution
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = ops.reshape(x, (b, h, w, c,))

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = self.roll_neg(x)
            # shifted_x = numpy.roll(x, (-self.shift_size, -self.shift_size), (1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = self.window_partition(shifted_x)  # nW*B, window_size, window_size, C
        x_windows = ops.reshape(x_windows,
                                (-1, self.window_size * self.window_size, c,))  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = ops.reshape(attn_windows, (-1, self.window_size, self.window_size, c,))
        shifted_x = self.window_reverse(attn_windows, self.window_size, h, w)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = self.roll_pos(shifted_x)
        else:
            x = shifted_x

        x = ops.reshape(x, (b, h * w, c,))

        # FFN
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class Roll(nn.Cell):
    def __init__(
        self,
        shift_size: int,
        shift_axis: Tuple[int] = (1, 2),
    ) -> None:
        super().__init__()
        self.shift_size = to_2tuple(shift_size)
        self.shift_axis = shift_axis

    def construct(self, x: Tensor) -> Tensor:
        x = numpy.roll(x, self.shift_size, self.shift_axis)
        return x


class PatchMerging(nn.Cell):
    """Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        input_resolution: Tuple[int],
        dim: int,
        norm_layer: Optional[nn.Cell] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim[0] if isinstance(dim, tuple) and len(dim) == 1 else dim
        # Default False
        self.reduction = nn.Dense(in_channels=4 * dim, out_channels=2 * dim, has_bias=False)
        self.norm = norm_layer([dim * 4, ])
        self.H, self.W = self.input_resolution
        self.H_2, self.W_2 = self.H // 2, self.W // 2
        self.H2W2 = int(self.H * self.W // 4)
        self.dim_mul_4 = int(dim * 4)
        self.H2W2 = int(self.H * self.W // 4)

    def construct(self, x: Tensor) -> Tensor:
        """
        x: B, H*W, C
        """
        b = x.shape[0]
        x = ops.reshape(x, (b, self.H_2, 2, self.W_2, 2, self.dim))
        x = ops.transpose(x, (0, 1, 3, 4, 2, 5))
        x = ops.reshape(x, (b, self.H2W2, self.dim_mul_4))
        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(nn.Cell):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Cell, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Cell | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Optional[float] = 0.0,
        norm_layer: Optional[nn.Cell] = nn.LayerNorm,
        downsample: Optional[nn.Cell] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.CellList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,  # TODO: 这里window_size//2的时候特别慢
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def construct(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class PatchEmbed(nn.Cell):
    """Image to Patch Embedding

    Args:
        image_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Cell, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[nn.Cell] = None,
    ) -> None:
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [image_size[0] // patch_size[0], image_size[1] // patch_size[1]]
        self.image_size = image_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size,
                              pad_mode="pad", has_bias=True, weight_init="TruncatedNormal")

        if norm_layer is not None:
            if isinstance(embed_dim, int):
                embed_dim = (embed_dim,)
            self.norm = norm_layer(embed_dim, epsilon=1e-5)
        else:
            self.norm = None

    def construct(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        # FIXME look at relaxing size constraints
        x = ops.reshape(self.proj(x), (b, self.embed_dim, -1))  # b Ph*Pw c
        x = ops.transpose(x, (0, 2, 1))

        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformer(nn.Cell):
    r"""SwinTransformer model class, based on
    `"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" <https://arxiv.org/pdf/2103.14030>`_

    Args:
        image_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Cell): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 96,
        depths: Optional[List[int]] = None,
        num_heads: Optional[List[int]] = None,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[int] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: Optional[nn.Cell] = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            image_size=image_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = Parameter(Tensor(np.zeros(1, num_patches, embed_dim), dtype=mstype.float32))

        self.pos_drop = nn.Dropout(keep_prob=1.0 - drop_rate)

        # stochastic depth
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.CellList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        self.norm = norm_layer([self.num_features, ], epsilon=1e-5)
        self.classifier = nn.Dense(in_channels=self.num_features,
                                   out_channels=num_classes, has_bias=True) if num_classes > 0 else Identity()
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(init.TruncatedNormal(sigma=0.02),
                                                      cell.weight.shape, cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(init.initializer(init.Zero(), cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(init.initializer(init.One(), cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(init.initializer(init.Zero(), cell.beta.shape, cell.beta.dtype))

    def no_weight_decay(self) -> None:
        return {"absolute_pos_embed"}

    def no_weight_decay_keywords(self) -> None:
        return {"relative_position_bias_table"}

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.classifier(x)
        return x

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # B L C
        x = ops.mean(ops.transpose(x, (0, 2, 1)), 2)  # B C 1
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def swin_tiny(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> SwinTransformer:
    """Get SwinTransformer tiny model.
    Refer to the base class 'models.SwinTransformer' for more details.
    """
    default_cfg = default_cfgs["swin_tiny"]
    model = SwinTransformer(image_size=224, patch_size=4, in_chans=in_channels, num_classes=num_classes,
                            embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7,
                            mlp_ratio=4., qkv_bias=True, qk_scale=None,
                            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                            norm_layer=nn.LayerNorm, ape=False, patch_norm=True, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
