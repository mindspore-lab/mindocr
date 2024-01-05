from typing import Optional, Tuple

import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from ...utils.misc import is_ms_version_2
from ..utils.attention_cells import MultiHeadAttention, PositionalEncoding, PositionwiseFeedForward

__all__ = ["MasterDecoder"]


class MasterDecoder(nn.Cell):
    """MASTER Decoder, based on
    `"MASTER: Multi-Aspect Non-local Network for Scene Text Recognition"
    <https://arxiv.org/abs/2205.00159>`_.

    Args:
        in_channels: Number of the input channels.
        out_channels: Number of the output channels.
        batch_max_length: The maximum length of the output. Default: 25.
        multi_heads_count: NUmber of heads in attention layer. Default: 8.
        stacks: Number of the blocks in the decoder. Default: 3.
        dropout: Dropout value in the positional encoding and other layers. Default: 0.0.
        feed_forward_size: Hidden dimension in the feed foward layer. Default: 2048.
        padding_symbol: The index of the padding symbol. Default: 2
        share_parameter: Whether to use the shared attention layer and feed foward layer.
            Default: False.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batch_max_length: int = 25,
        multi_heads_count: int = 8,
        stacks: int = 3,
        dropout: float = 0.0,
        feed_forward_size: int = 2048,
        padding_symbol: int = 2,
        share_parameter: bool = False,
    ) -> None:
        super().__init__()
        self.share_parameter = share_parameter
        self.batch_max_length = batch_max_length

        self.attention = nn.CellList(
            [
                MultiHeadAttention(multi_heads_count, in_channels, dropout)
                for _ in range(1 if share_parameter else stacks)
            ]
        )
        self.source_attention = nn.CellList(
            [
                MultiHeadAttention(multi_heads_count, in_channels, dropout)
                for _ in range(1 if share_parameter else stacks)
            ]
        )
        self.position_feed_forward = nn.CellList(
            [
                PositionwiseFeedForward(in_channels, feed_forward_size, dropout)
                for _ in range(1 if share_parameter else stacks)
            ]
        )
        self.position = PositionalEncoding(in_channels, dropout)
        self.stacks = stacks
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm([in_channels], epsilon=1e-6)
        self.embedding = nn.Embedding(out_channels, in_channels)
        self.sqrt_model_size = np.sqrt(in_channels)
        self.padding_symbol = padding_symbol
        self.generator = nn.Dense(in_channels, out_channels)

        # mask related
        if is_ms_version_2():
            self.tril = ops.tril
        else:
            self.tril = nn.Tril()

        self.argmax = ops.Argmax(axis=-1)

    def _generate_target_mask(self, targets: Tensor) -> Tensor:
        target_pad_mask = targets != self.padding_symbol
        target_pad_mask = target_pad_mask[:, None, :, None]
        target_pad_mask = ops.cast(target_pad_mask, ms.int32)
        target_length = targets.shape[1]
        target_sub_mask = self.tril(ops.ones((target_length, target_length), ms.int32))
        target_mask = ops.bitwise_and(target_pad_mask, target_sub_mask)
        return target_mask

    def _decode(
        self,
        inputs: Tensor,
        targets: Tensor,
        source_mask: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
    ) -> Tensor:
        targets = self.embedding(targets) * self.sqrt_model_size
        targets = self.position(targets)
        output = targets
        for i in range(self.stacks):
            if self.share_parameter:
                actual_i = i
            else:
                actual_i = 0

            normed_output = self.layer_norm(output)
            output = output + self.dropout(
                self.attention[actual_i](
                    normed_output, normed_output, normed_output, target_mask
                )
            )
            normed_output = self.layer_norm(output)
            output = output + self.dropout(
                self.source_attention[actual_i](
                    normed_output, inputs, inputs, source_mask
                )
            )
            normed_output = self.layer_norm(output)
            output = output + self.dropout(
                self.position_feed_forward[actual_i](normed_output)
            )
        output = self.layer_norm(output)
        output = self.generator(output)
        return output

    def construct(
        self, inputs: Tensor, targets: Optional[Tuple[Tensor, ...]] = None
    ) -> Tensor:
        N = inputs.shape[0]
        num_steps = self.batch_max_length + 1  # for <STOP> symbol

        if targets is not None:
            # training branch
            targets = targets[0]
            targets = targets[:, :-1]
            target_mask = self._generate_target_mask(targets)
            logits = self._decode(inputs, targets, target_mask=target_mask)
            return logits
        else:
            # inference branch
            # <GO> symbol
            targets = ops.zeros((N, 1), ms.int32)
            probs = list()

            for i in range(num_steps):
                target_mask = self._generate_target_mask(targets)
                probs_step = self._decode(inputs, targets, target_mask=target_mask)
                next_input = self.argmax(probs_step)
                targets = ops.concat([targets, next_input[:, i: i+1]], axis=1)
                probs.append(probs_step[:, i])

            probs = ops.stack(probs, axis=1)
            probs = ops.softmax(probs, axis=-1)
        return probs
