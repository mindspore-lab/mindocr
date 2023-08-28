import math

import numpy as np

import mindspore as ms
from mindspore import nn

from ..utils.abinet_layers import ABINetBlock, PositionalEncoding
from ..utils.abinet_layers import TransformerDecoder as ms_TransformerDecoder
from ..utils.abinet_layers import _default_tfmer_cfg

__all__ = ["ABINetHead"]


class ABINetHead(nn.Cell):
    def __init__(self, in_channels, batchsize=96):
        super().__init__()
        self.iter_size = 3
        self.batchsize = batchsize
        self.in_channels = in_channels  # In order to fit the mindocr framework, it is not actually used.
        self.alignment = BaseAlignment()
        self.language = BCNLanguage(self.batchsize)
        self.max_length = 26  # additional stop token

    def construct(self, v_res):
        # v_res = nout
        a_res = v_res
        all_l_res = []
        all_a_res = []
        for _ in range(self.iter_size):
            ms_softmax = nn.Softmax()
            tokens = ms_softmax(a_res["logits"])
            lengths = a_res["pt_lengths"]
            lengths = ms.ops.clip_by_value(lengths, 2, self.max_length)
            l_res = self.language(
                tokens, lengths
            )
            all_l_res.append(l_res)
            a_res = self.alignment(l_res["feature"], v_res["feature"])
            all_a_res.append(a_res)

        if not self.training:
            return a_res["logits"]

        return all_a_res, all_l_res, v_res


def _calculate_fan_in_and_fan_out(shape):
    """
    calculate fan_in and fan_out

    Args:
        shape (tuple): input shape.

    Returns:
        Tuple, a tuple with two elements, the first element is `n_in` and the second element is `n_out`.
    """
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("'fan_in' and 'fan_out' can not be computed for tensor with fewer than"
                         " 2 dimensions, but got dimensions {}.".format(dimensions))
    if dimensions == 2:  # Linear
        fan_in = shape[1]
        fan_out = shape[0]
    else:
        num_input_fmaps = shape[1]
        num_output_fmaps = shape[0]
        receptive_field_size = 1
        for i in range(2, dimensions):
            receptive_field_size *= shape[i]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


class BaseAlignment(ABINetBlock):
    def __init__(self):
        super().__init__()
        d_model = 512

        self.loss_weight = 1.0
        self.max_length = 26  # additional stop token
        self.w_att = nn.Dense(
            2 * d_model, d_model, weight_init='HeUniform', bias_init='uniform'
        )
        self.cls = nn.Dense(
            d_model,
            self.charset.num_classes, weight_init='HeUniform', bias_init='uniform'
        )
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                print("Dense Init HeUniform")
                cell.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.HeUniform(negative_slope=math.sqrt(5), mode="fan_in",
                                                    nonlinearity="leaky_relu"),
                    cell.weight.shape, cell.weight.dtype))
                weight = cell.weight
                fan_in, _ = _calculate_fan_in_and_fan_out(weight.shape)
                bound = 1 / math.sqrt(int(fan_in))

                cell.bias.set_data(ms.common.initializer.initializer(ms.common.initializer.Uniform(scale=bound),
                                                                     cell.bias.shape, cell.bias.dtype))

    def construct(self, l_feature, v_feature):

        f = ms.ops.concat((l_feature, v_feature), axis=2)

        f_att = ms.ops.sigmoid(self.w_att(f))

        output = f_att * v_feature + (1 - f_att) * l_feature
        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        return {
            "logits": logits,
            "pt_lengths": pt_lengths,
            "loss_weight": self.loss_weight,
            "name": "alignment",
        }


class BCNLanguage(ABINetBlock):
    def __init__(
        self, batchsize
    ):
        super().__init__()
        d_model = _default_tfmer_cfg["d_model"]
        nhead = _default_tfmer_cfg["nhead"]
        d_inner = _default_tfmer_cfg["d_inner"]
        dropout = _default_tfmer_cfg["dropout"]
        self.batchsize = batchsize
        num_layers = 4
        self.d_model = d_model
        self.detach = True
        self.use_self_attn = False
        self.loss_weight = 1.0
        self.max_length = 26  # additional stop token
        self.debug = False

        self.proj = nn.Dense(
            self.charset.num_classes,
            d_model,
            weight_init="uniform",
            bias_init="uniform",
            has_bias=False,
        )
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(
            d_model, dropout=1.0, max_len=self.max_length
        )
        self.model = ms_TransformerDecoder(
            batch_size=self.batchsize,
            num_layers=num_layers,
            hidden_size=self.d_model,
            num_heads=nhead,
            ffn_hidden_size=d_inner,
            hidden_dropout_rate=dropout,
            attention_dropout_rate=dropout,
            hidden_act="relu",
            src_seq_length=26,
            tgt_seq_length=26,
        )

        self.cls = nn.Dense(
            self.d_model,
            self.charset.num_classes,
            weight_init="uniform",
            bias_init="uniform",
        )

    def mindspore_decoder_mask(self, lengths):
        ms_unqueeze = ms.ops.expand_dims
        ms_pad_mask = self._get_padding_mask(lengths, 26)
        ms_pad_mask = ms_unqueeze(ms_pad_mask, -2)
        ms_eye_mask = self._get_location_mask(26)
        ms_eye_mask = ms_unqueeze(ms_eye_mask, 0)
        bitand = ms.ops.logical_and
        out_mask = bitand(ms_pad_mask, ms_eye_mask)

        return (out_mask).astype(ms.float16)

    def _get_padding_mask(self, length, max_length):
        ms_unqueeze = ms.ops.expand_dims
        length = ms_unqueeze(length, -1)
        grid = ms.numpy.arange(0, max_length)
        grid = ms_unqueeze(grid, 0)
        return grid < length

    def _get_location_mask(self, sz):
        a = np.eye(sz, sz)
        b = np.ones((26, 26))
        mask = b - a
        mask = ms.Tensor(mask)
        return mask.astype(ms.bool_)

    def construct(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        # if self.detach: tokens = tokens.detach()
        tokens1 = ms.ops.stop_gradient(tokens)
        embed = self.proj(tokens1)  # (N, T, E)
        embed = embed.transpose(1, 0, 2)
        embed = self.token_encoder(embed)  # (T, N, E)
        embed = embed.transpose(1, 0, 2)
        zeros = ms.ops.zeros((self.batchsize, 26, 512), ms.float32)
        zeros = zeros.transpose(1, 0, 2)
        query = self.pos_encoder(zeros)
        query = query.transpose(1, 0, 2)
        padding_mask = self.mindspore_decoder_mask(lengths)
        location_mask = self.mindspore_decoder_mask(lengths)
        output = self.model(query, padding_mask, embed, location_mask)
        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res = {
            "feature": output,
            "logits": logits,
            "pt_lengths": pt_lengths,
            "loss_weight": self.loss_weight,
            "name": "language",
        }

        return res
