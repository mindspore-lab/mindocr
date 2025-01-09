from typing import Optional, Tuple

import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

__all__ = ["MultiHeadAttention", "PositionwiseFeedForward", "PositionalEncoding", "SEModule"]


class MultiHeadAttention(nn.Cell):
    def __init__(
        self, multi_attention_heads: int, dimensions: int, dropout: float = 0.1
    ) -> None:
        """ """
        super(MultiHeadAttention, self).__init__()

        assert dimensions % multi_attention_heads == 0
        # requires d_v = d_k, d_q = d_k = d_v = d_m / h
        self.d_k = int(dimensions / multi_attention_heads)
        self.h = multi_attention_heads
        self.linears = nn.CellList([nn.Dense(dimensions, dimensions) for _ in range(4)])
        self.attention = None
        self.dropout = nn.Dropout(p=dropout)

        self.matmul = ops.BatchMatMul()

        self.min_fp16 = ms.tensor(np.finfo(np.float16).min, dtype=ms.float16)
        self.min_fp32 = ms.tensor(np.finfo(np.float32).min, dtype=ms.float32)
        self.min_fp64 = ms.tensor(np.finfo(np.float64).min, dtype=ms.float64)
        self.min_bf16 = ms.tensor(float.fromhex("-0x1.fe00000000000p+127"), dtype=ms.bfloat16)

    def dtype_to_min(self, dtype):
        if dtype == ms.float16:
            return self.min_fp16
        if dtype == ms.float32:
            return self.min_fp32
        if dtype == ms.float64:
            return self.min_fp64
        if dtype == ms.bfloat16:
            return self.min_bf16
        else:
            raise ValueError(f"Only support get minimum value of (float16, ), but got {dtype}")

    def dot_product_attention(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:

        d_k = float(value.shape[-1])
        d_k_sqrt = ops.cast(ms.numpy.sqrt(d_k), query.dtype)
        score = self.matmul(query, key.transpose(0, 1, 3, 2)) / d_k_sqrt  # (N, h, seq_len, seq_len)

        if mask is not None:
            score = ops.masked_fill(
                score, mask == 0, self.dtype_to_min(score.dtype)
            )  # score (N, h, seq_len, seq_len)

        p_attn = ops.softmax(score, axis=-1)
        # (N, h, seq_len, d_v), (N, h, seq_len, seq_len)
        return self.matmul(p_attn, value), p_attn

    def construct(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        N = query.shape[0]

        # do all the linear projections in batch from d_model => h x d_k
        # (N, seq_len, d_m) -> (N, seq_len, h, d_k) -> (N, h, seq_len, d_k)
        query = (
            self.linears[0](query)
            .reshape(N, -1, self.h, self.d_k)
            .transpose(0, 2, 1, 3)
        )
        key = (
            self.linears[1](key).reshape(N, -1, self.h, self.d_k).transpose(0, 2, 1, 3)
        )
        value = (
            self.linears[2](value)
            .reshape(N, -1, self.h, self.d_k)
            .transpose(0, 2, 1, 3)
        )

        # apply attention on all the projected vectors in batch.
        # (N, h, seq_len, d_v), (N, h, seq_len, seq_len)
        product_and_attention = self.dot_product_attention(query, key, value, mask=mask)
        x = product_and_attention[0]

        # "Concat" using a view and apply a final linear.
        # (N, seq_len, d_m)
        x = x.transpose(0, 2, 1, 3).reshape(N, -1, self.h * self.d_k)

        # (N, seq_len, d_m)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Cell):
    def __init__(
        self, dimensions: int, feed_forward_dimensions: int, dropout: float = 0.1
    ) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Dense(dimensions, feed_forward_dimensions)
        self.w_2 = nn.Dense(feed_forward_dimensions, dimensions)
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, input_tensor: Tensor) -> Tensor:
        return self.w_2(self.dropout(ops.relu(self.w_1(input_tensor))))


class PositionalEncoding(nn.Cell):
    def __init__(
        self, dimensions: int, dropout: float = 0.1, max_len: int = 5000
    ) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = np.zeros((max_len, dimensions), dtype=np.float32)
        position = np.arange(0, max_len)[..., None]
        div_term = np.exp(-np.arange(0, dimensions, 2) * np.log(10000) / dimensions)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[None, ...]
        self.pe = Tensor(pe, dtype=ms.float32)

    def construct(self, input_tensor: Tensor) -> Tensor:
        input_tensor = (
            input_tensor + self.pe[:, : input_tensor.shape[1]]
        )  # pe 1 5000 512
        return self.dropout(input_tensor)


class SEModule(nn.Cell):
    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            has_bias=True)
        self.conv2 = nn.Conv2d(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            has_bias=True)
        self.relu = nn.ReLU()
        self.h_sigmoid = nn.HSigmoid()

    def construct(self, inputs):
        outputs = ops.mean(inputs, axis=(-2, -1), keep_dims=True)  # equivalent to nn.AdaptiveAvgPool2d(1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.h_sigmoid(outputs)
        return inputs * outputs
