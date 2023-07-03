from typing import Optional, Tuple

import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

__all__ = ["MultiHeadAttention", "PositionwiseFeedForward", "PositionalEncoding"]


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
        self.dropout = nn.Dropout(keep_prob=1 - dropout)

        self.matmul = ops.BatchMatMul()

    def dot_product_attention(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:

        d_k = value.shape[-1]
        score = self.matmul(query, key.transpose(0, 1, 3, 2)) / ms.numpy.sqrt(
            d_k
        )  # (N, h, seq_len, seq_len)

        # cast to fp32 to prevent overflow
        score = ops.cast(score, ms.float32)

        if mask is not None:
            score = ops.masked_fill(
                score, mask == 0, -1e9
            )  # score (N, h, seq_len, seq_len)

        p_attn = ops.softmax(score, axis=-1)
        p_attn = ops.cast(p_attn, value.dtype)
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
        self.dropout = nn.Dropout(keep_prob=1 - dropout)

    def construct(self, input_tensor: Tensor) -> Tensor:
        return self.w_2(self.dropout(ops.relu(self.w_1(input_tensor))))


class PositionalEncoding(nn.Cell):
    def __init__(
        self, dimensions: int, dropout: float = 0.1, max_len: int = 5000
    ) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(keep_prob=1 - dropout)

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
