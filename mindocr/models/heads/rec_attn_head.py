from typing import Tuple, Optional

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from ..utils import GRUCell


__all__ = ["AttentionHead"]


class AttentionHead(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int = 256,
        batch_max_length: int = 25,
    ) -> None:
        """
        Inputs:
            inputs: shape [W, BS, 2*C]
            label: shape [BS, W]
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels
        self.batch_max_length = batch_max_length

        self.attention_cell = AttentionCell(
            self.in_channels, self.hidden_size, self.num_classes
        )
        self.generator = nn.Dense(hidden_size, self.num_classes)

        self.one = Tensor(1.0, ms.float32)
        self.zero = Tensor(0.0, ms.float32)

    def _char_to_onehot(self, input_char: Tensor, onehot_dim: int) -> Tensor:
        input_one_hot = ops.one_hot(input_char, onehot_dim, self.one, self.zero)
        return input_one_hot

    def construct(self, inputs: Tensor, targets: Optional[Tensor] = None) -> Tensor:
        # convert the inputs from [W, BS, C] to [BS, W, C]
        inputs = ops.transpose(inputs, (1, 0, 2))
        N = inputs.shape[0]
        num_steps = self.batch_max_length + 1  # for <STOP> symbol

        hidden = ops.zeros((N, self.hidden_size), inputs.dtype)

        if targets is not None:
            # training branch
            output_hiddens = list()
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets[:, i], self.num_classes)
                hidden, _ = self.attention_cell(hidden, inputs, char_onehots)
                output_hiddens.append(ops.expand_dims(hidden, axis=1))
            output = ops.concat(output_hiddens, axis=1)
            probs = self.generator(output)
        else:
            # inference branch
            # <GO> symbol
            targets = ops.zeros((N,), ms.int32)
            probs = list()
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, self.num_classes)
                hidden, _ = self.attention_cell(hidden, inputs, char_onehots)
                probs_step = self.generator(hidden)
                probs.append(probs_step)
                next_input = ops.argmax(probs_step, axis=1)
                targets = next_input
            probs = ops.stack(probs, axis=1)
            probs = ops.softmax(probs, axis=2)
        return probs


class AttentionCell(nn.Cell):
    def __init__(self, input_size: int, hidden_size: int, num_embeddings: int) -> None:
        super().__init__()
        self.i2h = nn.Dense(input_size, hidden_size, has_bias=False)
        self.h2h = nn.Dense(hidden_size, hidden_size)
        self.score = nn.Dense(hidden_size, 1, has_bias=False)
        self.rnn = GRUCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

        self.bmm = ops.BatchMatMul()

    def construct(
        self, prev_hidden: Tensor, batch_H: Tensor, char_onehots: Tensor
    ) -> Tuple[Tensor, Tensor]:
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden)
        prev_hidden_proj = ops.expand_dims(prev_hidden_proj, 1)

        res = ops.add(batch_H_proj, prev_hidden_proj)
        res = ops.tanh(res)
        e = self.score(res)

        alpha = ops.softmax(e, axis=1)
        alpha = ops.transpose(alpha, (0, 2, 1))
        context = ops.squeeze(self.bmm(alpha, batch_H), axis=1)
        concat_context = ops.concat([context, char_onehots], 1)

        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha
