from typing import Optional, Tuple

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.numpy as np
import mindspore.ops as ops
from mindspore import Parameter, nn

from mindocr.nlp.utils.layers import LayerNorm, Linear


class MLPBlock(nn.Cell):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: nn.Cell = nn.GELU,
        compute_dtype=mstype.float16,
        param_init_type=mstype.float32,
    ) -> None:
        super().__init__()
        self.lin1 = Linear(
            in_channels=embedding_dim,
            out_channels=mlp_dim,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type,
        )
        self.lin2 = Linear(
            in_channels=mlp_dim,
            out_channels=embedding_dim,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type,
        )
        self.act = act()

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Cell):
    """
    Layer Normalization for 2D data.

    Inputs:
        x (ms.Tensor): Input tensor.

    Returns:
        ms.Tensor: Normalized tensor.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6, param_init_type=ms.float32) -> None:
        super().__init__()
        self.weight = Parameter(ops.Ones()(num_channels, param_init_type))
        self.bias = Parameter(ops.Zeros()(num_channels, param_init_type))
        self.eps = eps
        self.pow = ops.Pow()
        self.sqrt = ops.Sqrt()

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        u = x.mean(1, keep_dims=True)
        s = self.pow(x - u, 2).mean(1, keep_dims=True)
        x = (x - u) / self.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SAMImageEncoder(nn.Cell):
    """
    Image encoder
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.in_chans = config.in_chans
        self.embed_dim = config.embed_dim
        self.depth = config.depth
        self.num_heads = config.num_heads
        self.mlp_ratio = config.mlp_ratio
        self.out_chans = config.out_chans
        self.qkv_bias = config.qkv_bias
        self.layer_norm_eps = config.layer_norm_eps
        self.use_abs_pos = config.use_abs_pos
        self.use_rel_pos = config.use_rel_pos
        self.window_size = config.window_size
        self.global_attn_indexes = config.global_attn_indexes

        self.compute_dtype = config.compute_dtype
        self.layernorm_compute_type = config.layernorm_compute_type
        self.softmax_compute_type = config.softmax_compute_type
        self.param_init_type = config.param_init_type
        self.ln_param_init_type = config.ln_param_init_type

        if isinstance(self.img_size, int):
            img_h = self.img_size
            img_w = self.img_size
        else:
            img_h, img_w = self.img_size
        feat_h = img_h // self.patch_size
        feat_w = img_w // self.patch_size
        self.feat_size = (feat_h, feat_w)
        if self.window_size > 0:
            pad_h = (self.window_size - feat_h % self.window_size) % self.window_size
            pad_w = (self.window_size - feat_w % self.window_size) % self.window_size
            self.pad_size = (pad_h, pad_w)
        else:
            self.pad_size = (0, 0)

        self.patch_embed = PatchEmbed(
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            param_init_type=self.param_init_type,
        )

        self.pos_embed: Optional[Parameter] = None
        if self.use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = Parameter(
                ops.Zeros()(
                    (1, self.img_size // self.patch_size, self.img_size // self.patch_size, self.embed_dim),
                    self.param_init_type,
                )
            )

        self.blocks = nn.CellList()
        for i in range(self.depth):
            block = Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                use_rel_pos=self.use_rel_pos,
                window_size=self.window_size if i not in self.global_attn_indexes else 0,
                pad_size=self.pad_size,
                feat_size=self.feat_size,
                input_size=(self.img_size // self.patch_size, self.img_size // self.patch_size),
                layer_norm_eps=self.layer_norm_eps,
                compute_dtype=self.compute_dtype,
                layernorm_compute_type=self.layernorm_compute_type,
                softmax_compute_type=self.softmax_compute_type,
                param_init_type=self.param_init_type,
                ln_param_init_type=self.ln_param_init_type,
            )
            self.blocks.append(block)

        self.neck = nn.SequentialCell(
            nn.Conv2d(
                self.embed_dim,
                self.out_chans,
                kernel_size=1,
                has_bias=False,
                dtype=self.param_init_type,
            ),
            LayerNorm2d(self.out_chans, param_init_type=self.ln_param_init_type),
            nn.Conv2d(
                self.out_chans,
                self.out_chans,
                kernel_size=3,
                pad_mode="pad",
                padding=1,
                has_bias=False,
                dtype=self.param_init_type,
            ),
            LayerNorm2d(self.out_chans, param_init_type=self.ln_param_init_type),
        )

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """
        Args:
            x (ms.Tensor): Input image tensor.

        Returns:
            ms.Tensor: Encoded image tensor.
        """
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.transpose(0, 3, 1, 2))

        return x


class Block(nn.Cell):
    """
    Transformer blocks with support of window attention and residual propagation blocks
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        window_size: int = 0,
        pad_size: Tuple[int, int] = (0, 0),
        feat_size: Tuple[int, int] = (64, 64),
        input_size: Optional[Tuple[int, int]] = None,
        layer_norm_eps: float = 1.0e-12,
        compute_dtype=mstype.float16,
        layernorm_compute_type=mstype.float32,
        softmax_compute_type=mstype.float32,
        param_init_type=mstype.float32,
        ln_param_init_type=mstype.float32,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm((dim,), eps=layer_norm_eps, param_init_type=ln_param_init_type)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            compute_dtype=compute_dtype,
            layernorm_compute_type=layernorm_compute_type,
            softmax_compute_type=softmax_compute_type,
            param_init_type=param_init_type,
        )

        self.norm2 = LayerNorm((dim,), eps=layer_norm_eps, param_init_type=ln_param_init_type)
        self.mlp = MLPBlock(
            embedding_dim=dim,
            mlp_dim=int(dim * mlp_ratio),
            compute_dtype=compute_dtype,
            param_init_type=param_init_type,
        )

        self.window_size = window_size
        self.pad_size = pad_size
        self.feat_size = feat_size

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """
        Args:
            x (ms.Tensor): Input tensor.

        Returns:
            ms.Tensor: Output tensor.
        """
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            pad_size = self.pad_size
            window_size = self.window_size
            b, h, w, c = x.shape
            pad_h, pad_w = pad_size
            if pad_h > 0 or pad_w > 0:
                pad = ops.Pad(paddings=((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
                x = pad(x)
            hp, wp = h + pad_h, w + pad_w

            x = x.view(b, hp // window_size, window_size, wp // window_size, window_size, c)
            x = x.transpose(0, 1, 3, 2, 4, 5).view(-1, window_size, window_size, c)
            # x = window_partition(x, self.window_size, self.pad_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, self.pad_size, self.feat_size)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Cell):
    """
    Multi-head Attention block with relative position embeddings.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
        compute_dtype=mstype.float16,
        layernorm_compute_type=mstype.float32,
        softmax_compute_type=mstype.float32,
        param_init_type=mstype.float32,
    ) -> None:
        super().__init__()
        self.compute_dtype = compute_dtype
        self.layernorm_compute_type = layernorm_compute_type
        self.softmax_compute_type = softmax_compute_type

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = Linear(
            in_channels=dim,
            out_channels=dim * 3,
            has_bias=qkv_bias,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type,
        )
        self.proj = Linear(
            in_channels=dim,
            out_channels=dim,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type,
        )

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = Parameter(ops.Zeros()((2 * input_size[0] - 1, head_dim), self.compute_dtype))
            self.rel_pos_w = Parameter(ops.Zeros()((2 * input_size[1] - 1, head_dim), self.compute_dtype))

        self.softmax = ops.Softmax(axis=-1)
        self.batchmatmul = ops.BatchMatMul()
        self.batchmatmul_trans_b = ops.BatchMatMul(transpose_b=True)
        self.cast = ops.Cast()
        self.unstack = ops.Unstack(axis=0)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """
        Args:
            x (ms.Tensor): Input tensor.

        Returns:
            ms.Tensor: Output tensor.
        """
        b, h, w, _ = x.shape
        ori_type = x.dtype
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(b, h * w, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        qkv = self.cast(qkv, self.compute_dtype)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = self.unstack(qkv.reshape(3, b * self.num_heads, h * w, -1))

        attn = self.batchmatmul_trans_b((q * self.scale), k)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (h, w), (h, w))

        attn = self.cast(attn, self.softmax_compute_type)
        attn = self.softmax(attn)
        attn = self.cast(attn, self.compute_dtype)
        x = self.batchmatmul(attn, v)
        x = x.view(b, self.num_heads, h, w, -1)
        x = x.permute(0, 2, 3, 1, 4)
        x = x.reshape(b, h, w, -1)
        x = self.proj(x)
        x = self.cast(x, ori_type)

        return x


def window_partition(x: ms.Tensor, window_size: int, pad_size: Tuple[int, int] = 0) -> ms.Tensor:
    """
    Partition the input tensor into non-overlapping windows with optional padding.

    Args:
        x (ms.Tensor): Input tensor with shape [B, H, W, C].
        window_size (int): Window size.
        pad_size (tuple[int, int]): Padding size as (pad_h, pad_w).

    Returns:
        windows (ms.Tensor): Windows after partition with shape [B * num_windows, window_size, window_size, C].
    """
    b, h, w, c = x.shape
    pad_h, pad_w = pad_size
    if pad_h > 0 or pad_w > 0:
        pad = ops.Pad(paddings=((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
        x = pad(x)
    hp, wp = h + pad_h, w + pad_w

    x = x.view(b, hp // window_size, window_size, wp // window_size, window_size, c)
    windows = x.transpose(0, 1, 3, 2, 4, 5).view(-1, window_size, window_size, c)
    return windows


def window_unpartition(windows: ms.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]) -> ms.Tensor:
    """
    Unpartition windows back into original sequences and remove padding if needed.

    Args:
        windows (ms.Tensor): Input windows with shape [B * num_windows, window_size, window_size, C].
        window_size (int): Window size.
        pad_hw (Tuple[int, int]): Padded height and width (Hp, Wp).
        hw (Tuple[int, int]): Original height and width (H, W) before padding.

    Returns:
        x (ms.Tensor): Unpartitioned sequences with shape [B, H, W, C].
    """
    pad_h, pad_w = pad_hw
    h, w = hw
    hp, wp = h + pad_h, w + pad_w
    b = windows.shape[0] // (hp * wp // window_size // window_size)
    x = windows.view(b, hp // window_size, wp // window_size, window_size, window_size, -1)
    x = x.transpose(0, 1, 3, 2, 4, 5).view(b, hp, wp, -1)

    if hp > h or wp > w:
        x = x[:, :h, :w, :]
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: ms.Tensor) -> ms.Tensor:
    """
    Get relative positional embeddings based on the relative positions of query and key sizes.

    Args:
        q_size (int): Size of query q.
        k_size (int): Size of key k.
        rel_pos (ms.Tensor): Relative position embeddings (L, C).

    Returns:
        ms.Tensor: Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = ops.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).transpose(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = np.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = np.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.astype(mstype.int32)]


def add_decomposed_rel_pos(
    attn: ms.Tensor,
    q: ms.Tensor,
    rel_pos_h: ms.Tensor,
    rel_pos_w: ms.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> ms.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from mvitv2 paper.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950

    Args:
        attn (ms.Tensor): Attention map.
        q (ms.Tensor): Query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (ms.Tensor): Relative position embeddings (Lh, C) for the height axis.
        rel_pos_w (ms.Tensor): Relative position embeddings (Lw, C) for the width axis.
        q_size (Tuple[int, int]): Spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple[int, int]): Spatial sequence size of key k with (k_h, k_w).

    Returns:
        ms.Tensor: Attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    rh = get_rel_pos(q_h, k_h, rel_pos_h)
    rw = get_rel_pos(q_w, k_w, rel_pos_w)

    b, _, dim = q.shape
    r_q = q.reshape(b, q_h, q_w, dim)
    rel_h = ops.matmul(r_q, rh.transpose(0, 2, 1)).reshape(b, q_h, q_w, rh.shape[1])
    rel_w = ops.matmul(r_q.transpose(0, 2, 1, 3), rw.transpose(0, 2, 1)).reshape(b, q_h, q_w, rw.shape[1])
    rel_w = rel_w.transpose(0, 2, 1, 3)

    attn = (attn.view(b, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(
        b, q_h * q_w, k_h * k_w
    )

    return attn


class PatchEmbed(nn.Cell):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
        param_init_type=ms.float32,
    ) -> None:
        """
        Initialize the PatchEmbed layer.

        Args:
            kernel_size (Tuple[int, int]): Kernel size of the projection layer.
            stride (Tuple[int, int]): Stride of the projection layer.
            padding (Tuple[int, int, int, int]): Padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            has_bias=True,
            dtype=param_init_type,
        )

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """
        Forward pass of the PatchEmbed layer.

        Args:
            x (ms.Tensor): Input image tensor.

        Returns:
            ms.Tensor: Patch embeddings tensor.
        """
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.transpose(0, 2, 3, 1)
        return x


class SAMEncoder(SAMImageEncoder):
    """SAM encoder for Vary system"""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.net_2 = nn.Conv2d(
            256, 512, kernel_size=3, stride=2, pad_mode="pad", padding=1, has_bias=False, dtype=config.param_init_type
        )
        self.net_3 = nn.Conv2d(
            512, 1024, kernel_size=3, stride=2, pad_mode="pad", padding=1, has_bias=False, dtype=config.param_init_type
        )

    def construct(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.transpose(0, 3, 1, 2))

        x = self.net_2(x)
        x = self.net_3(x)
        x = x.flatten(start_dim=2).permute(0, 2, 1)
        return x
