import math

import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import _checkparam as Validator
from mindspore import log as logger
from mindspore import nn
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.log import _LogActionOnce
from mindspore.nn.cell import Cell
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.primitive import constexpr
from mindspore.parallel._transformer.layers import (
    _args_type_validator_check,
    _check_input_dtype,
    _check_past_none_input_none,
    _LayerInputCheck,
    _LayerNorm,
    _valid_type_checks,
    _valid_value_checks,
)
from mindspore.parallel._transformer.moe import MoE, _check_moe_config, default_moe_config
from mindspore.parallel._transformer.op_parallel_config import (
    MoEParallelConfig,
    OpParallelConfig,
    _check_config,
    default_dpmp_config,
)
from mindspore.parallel._transformer.transformer import (
    FeedForward,
    MultiHeadAttention,
    TransformerOpParallelConfig,
    _get_lambda_func,
    default_transformer_config,
)
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

_default_tfmer_cfg = dict(
    d_model=512, nhead=8, d_inner=2048, dropout=0.1, activation="relu"  # 1024
)


@constexpr
def _check_shape_equal(input_shape, param_name, func_name, target_shape):
    _LayerInputCheck.check_shape_equal(input_shape, param_name, func_name, target_shape)


class ABINetBlock(nn.Cell):
    def __init__(self):
        super().__init__()
        self.max_length = 26
        self.charset = CharsetMapper(
            max_length=self.max_length,
        )

    def _get_length(self, logit, dim=-1):

        logit_argmax = ms.ops.Argmax()(logit)
        out = logit_argmax == 0
        out_copy = out.copy()
        abn = out.any(dim)
        out1 = out.cumsum(dim) == 1
        out = ms.ops.logical_and(out_copy, out1)
        out1 = out.argmax(-1)
        out1 = out1 + 1
        logit_shape1 = logit.shape[1]
        out = ms.numpy.where(abn, out1, logit_shape1)
        return out

    @staticmethod
    def _get_padding_mask(length, max_length):
        length = ms.numpy.expand_dims(length, -1)
        # length = length.unsqueeze(-1)
        grid = ms.numpy.arange(0, max_length)
        grid = ms.numpy.expand_dims(grid, 0)
        grid = ms.Tensor(grid)
        return grid >= length

    @staticmethod
    def _get_location_mask(sz, device=None):
        eyes = ms.ops.Eye()
        mask1 = eyes(sz, sz, ms.bool_)
        cast = ms.ops.Cast()
        mask = cast(mask1, ms.float32)
        mask = ms.ops.masked_fill(mask, mask1, float("-inf"))
        expand_dims = ms.ops.ExpandDims()
        mask = expand_dims(mask, 0)
        # mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask


class CharsetMapper(object):

    def __init__(self, max_length=30, null_char="\u2591"):

        self.null_char = null_char
        self.max_length = max_length
        self.label_to_char = self._read_charset()
        self.char_to_label = dict(map(reversed, self.label_to_char.items()))
        self.num_classes = len(self.label_to_char)

    def _read_charset(self):
        charset = {}
        charset_list = "░abcdefghijklmnopqrstuvwxyz1234567890"
        charset = {idx: c for idx, c in enumerate(charset_list)}
        self.null_label = 0
        charset[self.null_label] = self.null_char
        return charset

    def trim(self, text):
        assert isinstance(text, str)
        return text.replace(self.null_char, "")

    def get_text(self, labels, length=None, padding=True, trim=False):
        """Returns a string corresponding to a sequence of character ids."""
        length = length if length else self.max_length
        labels = [int(a) if isinstance(a, ms.Tensor) else int(a) for a in labels]
        if padding:
            labels = labels + [self.null_label] * (length - len(labels))
        text = "".join([self.label_to_char[label] for label in labels])
        if trim:
            text = self.trim(text)
        return text

    def get_labels(self, text, length=None, padding=True, case_sensitive=False):
        """Returns the labels of the corresponding text."""
        length = length if length else self.max_length
        if padding:
            text = text + self.null_char * (length - len(text))
        if not case_sensitive:
            text = text.lower()
        labels = [self.char_to_label[char] for char in text]
        return labels

    def pad_labels(self, labels, length=None):
        length = length if length else self.max_length
        return labels + [self.null_label] * (length - len(labels))

    @property
    def digits(self):
        return "0123456789"

    @property
    def digit_labels(self):
        return self.get_labels(self.digits, padding=False)

    @property
    def alphabets(self):
        all_chars = list(self.char_to_label.keys())
        valid_chars = []
        for c in all_chars:
            if c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
                valid_chars.append(c)
        return "".join(valid_chars)

    @property
    def alphabet_labels(self):
        return self.get_labels(self.alphabets, padding=False)


def onehot(label, depth, device=None):

    label_shape = 26

    onehot_output = np.zeros((label_shape, depth))

    label_expand = np.expand_dims(label, -1)
    label_expand = np.expand_dims(label_expand, -1)
    label_expand_onehot = np.zeros((26, 37))
    a = 0
    for i in label_expand:
        i = int(i)
        label_expand_onehot[a][i] = 1
        a = a + 1

    label_expand_onehot = label_expand_onehot
    onehot_output = label_expand_onehot + onehot_output

    return onehot_output


def encoder_layer(in_c, out_c, k=3, s=2, p=1):
    return nn.SequentialCell(
        nn.Conv2d(in_c, out_c, k, s, pad_mode="pad", padding=p, has_bias=True),
        nn.BatchNorm2d(out_c, momentum=0.1),
        nn.ReLU(),
    )


class ms_upsample_scale(nn.Cell):
    def __init__(self, scale_factor, align_corners):
        super().__init__()
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def construct(self, x):
        _1, _2, height, width = x.shape
        new_height = self.scale_factor * height
        new_width = self.scale_factor * width
        resize = ms.ops.ResizeNearestNeighbor(
            size=(new_height, new_width), align_corners=self.align_corners
        )
        x = resize(x)
        return x


class ms_upsample_size(nn.Cell):
    def __init__(self, size, align_corners):
        super().__init__()
        self.size = size
        self.align_corners = align_corners

    def construct(self, x):
        resize = ms.ops.ResizeNearestNeighbor(
            size=self.size, align_corners=self.align_corners
        )
        x = resize(x)
        return x


# mindspore upsample ResizeBilinear 只有bilinear
def decoder_layer1(
    in_c, out_c, k=3, s=1, p=1, mode="nearest", scale_factor=None, size=None
):
    align_corners = False if mode == "nearest" else True
    return nn.SequentialCell(
        ms_upsample_scale(scale_factor, align_corners=align_corners),
        nn.Conv2d(in_c, out_c, k, s, pad_mode="pad", padding=p, has_bias=True),
        nn.BatchNorm2d(out_c, momentum=0.1),
        nn.ReLU(),
    )


def decoder_layer2(
    in_c, out_c, k=3, s=1, p=1, mode="nearest", scale_factor=None, size=None
):
    align_corners = False if mode == "nearest" else True
    return nn.SequentialCell(
        ms_upsample_size(size, align_corners=align_corners),
        nn.Conv2d(in_c, out_c, k, s, pad_mode="pad", padding=p, has_bias=True),
        nn.BatchNorm2d(out_c, momentum=0.1),
        nn.ReLU(),
    )


class PositionAttention(nn.Cell):
    def __init__(
        self,
        max_length,
        in_channels=512,
        num_channels=64,
        h=8,
        w=32,
        mode="nearest",
        **kwargs
    ):
        super().__init__()
        self.max_length = max_length
        self.k_encoder1 = encoder_layer(in_channels, num_channels, s=(1, 2))
        self.k_encoder2 = encoder_layer(num_channels, num_channels, s=(2, 2))
        self.k_encoder3 = encoder_layer(num_channels, num_channels, s=(2, 2))
        self.k_encoder4 = encoder_layer(num_channels, num_channels, s=(2, 2))

        self.k_decoder1 = decoder_layer1(
            num_channels, num_channels, scale_factor=2, mode=mode
        )
        self.k_decoder2 = decoder_layer1(
            num_channels, num_channels, scale_factor=2, mode=mode
        )
        self.k_decoder3 = decoder_layer1(
            num_channels, num_channels, scale_factor=2, mode=mode
        )
        self.k_decoder4 = decoder_layer2(
            num_channels, in_channels, size=(h, w), mode=mode
        )

        self.pos_encoder = PositionalEncoding(
            in_channels, dropout=1.0, max_len=max_length
        )
        self.project = nn.Dense(
            in_channels, in_channels, weight_init="HeUniform", bias_init="uniform"
        )

    def construct(self, x):
        N, E, H, W = x.shape
        k, v = x, x  # (N, E, H, W)
        features = []
        k = self.k_encoder1(k)
        features.append(k)
        k = self.k_encoder2(k)
        features.append(k)
        k = self.k_encoder3(k)
        features.append(k)
        k = self.k_encoder4(k)
        features.append(k)
        k = self.k_decoder1(k)
        k = k + features[2]
        k = self.k_decoder2(k)
        k = k + features[1]
        k = self.k_decoder3(k)
        k = k + features[0]
        k = self.k_decoder4(k)

        k_1, k_2, k_3, k_4 = k.shape
        # calculate query vector
        # TODO q=f(q,k)
        zeros = ms.ops.Zeros()
        x_zeros = zeros((self.max_length, N, E), ms.float32)  # (T, N, E)
        q = self.pos_encoder(x_zeros)  # (T, N, E)
        q = q.transpose(1, 0, 2)
        q = self.project(q)  # (N, T, E)

        # calculate attention
        k_attn = k.view(k_1, k_2, -1)
        batmatmul = ms.ops.BatchMatMul()
        attn_scores = batmatmul(q, k_attn)  # (N, T, (H*W))
        attn_scores = attn_scores / (E**0.5)
        softmax_attn = nn.Softmax()
        attn_scores = softmax_attn(attn_scores)
        v = v.transpose(0, 2, 3, 1)
        v = v.view(N, -1, E)  # (N, (H*W), E)
        attn_vecs = batmatmul(attn_scores, v)  # (N, T, E)

        return attn_vecs, attn_scores.view(N, -1, H, W)


class PositionalEncoding(nn.Cell):
    def __init__(self, d_model=512, dropout=0.9, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=1 - dropout)
        pe = np.zeros((max_len, d_model), np.float32)
        position = np.arange(0, max_len, dtype=np.float32)
        position = np.expand_dims(position, 1)
        div = np.arange(0, d_model, 2, dtype=np.float32)
        div = div * (-math.log(10000.0) / d_model)
        div_term = np.exp(div)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = np.expand_dims(pe, 0)
        pe = np.transpose(pe, (1, 0, 2))
        pe = ms.Tensor(pe).astype(dtype=ms.float32)
        self.pe = ms.Parameter(pe, name="pe1", requires_grad=False)

    def construct(self, x):
        w, _, = x.shape
        x = x + self.pe[:w, :]

        return self.dropout(x)


# Since Mindspore Transformer does not meet the requirements
# It has been modified


class TransformerEncoderLayer(Cell):

    @_LogActionOnce(
        logger=logger,
        key="TransformerEncoderLayer",
        no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,),
    )
    @_args_type_validator_check(
        batch_size=Validator.check_positive_int,
        hidden_size=Validator.check_positive_int,
        num_heads=Validator.check_positive_int,
        ffn_hidden_size=Validator.check_positive_int,
        seq_length=Validator.check_positive_int,
        attention_dropout_rate=Validator.check_non_negative_float,
        hidden_dropout_rate=Validator.check_non_negative_float,
        hidden_act=_valid_type_checks([str], "TransformerEncoderLayer"),
        post_layernorm_residual=Validator.check_bool,
        layernorm_compute_type=_valid_value_checks(
            [mstype.float32, mstype.float16], "TransformerEncoderLayer"
        ),
        softmax_compute_type=_valid_value_checks(
            [mstype.float32, mstype.float16], "TransformerEncoderLayer"
        ),
        param_init_type=_valid_value_checks(
            [mstype.float32, mstype.float16], "TransformerEncoderLayer"
        ),
        parallel_config=_valid_type_checks(
            [OpParallelConfig, MoEParallelConfig], "TransformerEncoderLayer"
        ),
        use_past=Validator.check_bool,
    )
    def __init__(
        self,
        batch_size,
        hidden_size,
        ffn_hidden_size,
        num_heads,
        seq_length,
        attention_dropout_rate=0.1,
        hidden_dropout_rate=0.1,
        post_layernorm_residual=False,
        layernorm_compute_type=mstype.float32,
        softmax_compute_type=mstype.float32,
        param_init_type=mstype.float32,
        hidden_act="gelu",
        use_past=False,
        moe_config=default_moe_config,
        parallel_config=default_dpmp_config,
    ):
        super(TransformerEncoderLayer, self).__init__()
        if (
            _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,)
            and _is_sharding_propagation()
        ):
            _check_config(parallel_config)
            if num_heads % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'num_heads' must be divisibled by the "
                    "'parallel_config.model_parallel', but got the num_heads is {} and "
                    "parallel_config.model_parallel is {}.".format(
                        num_heads, parallel_config.model_parallel
                    )
                )
            if hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'hidden_size' must be divisibled by "
                    "the 'parallel_config.model_parallel', but got the hidden_size is {} and parallel_config."
                    " model_parallel is {}.".format(
                        hidden_size, parallel_config.model_parallel
                    )
                )
            if ffn_hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'ffn_hidden_size' must be divisibled "
                    "by the 'parallel_config.model_parallel', but got the ffn_hidden_size is {} "
                    "and parallel_config. model_parallel is {}.".format(
                        ffn_hidden_size, parallel_config.model_parallel
                    )
                )
            _check_moe_config(moe_config, parallel_config)
            self.use_moe = moe_config.expert_num > 1
            self.use_past = use_past
            self.seq_length = seq_length
            self.hidden_size = hidden_size
            self.batch_size = batch_size
            self.layernorm1 = _LayerNorm((hidden_size,)).to_float(
                layernorm_compute_type
            )
            self.layernorm2 = _LayerNorm((hidden_size,)).to_float(
                layernorm_compute_type
            )
            parallel_config_args = (
                parallel_config.dpmp if self.use_moe else parallel_config
            )
            self.attention = MultiHeadAttention(
                batch_size=batch_size,
                src_seq_length=seq_length,
                tgt_seq_length=seq_length,
                hidden_size=hidden_size,
                num_heads=num_heads,
                hidden_dropout_rate=hidden_dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                softmax_compute_type=softmax_compute_type,
                param_init_type=param_init_type,
                use_past=use_past,
                parallel_config=parallel_config_args,
            )
            # For ABINet, the following paragraph is deleted
            # if self.use_moe:
            #     self.output = MoE(hidden_size=hidden_size,
            #                       dropout_rate=hidden_dropout_rate,
            #                       ffn_hidden_size=ffn_hidden_size,
            #                       param_init_type=param_init_type,
            #                       hidden_act=hidden_act,
            #                       moe_config=moe_config,
            #                       parallel_config=parallel_config)
            # else:
            # Feed Forward Network, FFN
            self.output = FeedForward(
                hidden_size=hidden_size,
                dropout_rate=hidden_dropout_rate,
                ffn_hidden_size=ffn_hidden_size,
                param_init_type=param_init_type,
                hidden_act=hidden_act,
                parallel_config=parallel_config,
            )
            self.post_layernorm_residual = post_layernorm_residual
            self.add = P.Add().shard(
                ((parallel_config.data_parallel, 1), (parallel_config.data_parallel, 1))
            )
            self.add_3d = P.Add().shard(
                (
                    (parallel_config.data_parallel, 1, 1),
                    (parallel_config.data_parallel, 1, 1),
                )
            )
            self.dtype = mstype.float16
            self.key_past = None
            self.value_past = None

            # For ABINet, the following paragraph is deleted
            # if self.use_past:
            #     # operator used for state reuse
            #     self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
            #     self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
            #     self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
            #     size_per_head = int(hidden_size / num_heads)
            #     self.key_shape = (batch_size, num_heads, size_per_head, seq_length)
            #     self.value_shape = (batch_size, num_heads, seq_length, size_per_head)
            #     # parameters saving key and value states
            #     self.key_past = Parameter(Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past")
            #     self.value_past = Parameter(Tensor(np.zeros(shape=self.value_shape), self.dtype), name="value_past")
            #     self.tile = P.Tile().shard(((1, 1),))
            #     self.mul = P.Mul().shard(((1, 1, 1, 1), (1,)))
            #     self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        elif _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            _check_config(parallel_config)
            if num_heads % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'num_heads' must be divisibled by the "
                    "'parallel_config.model_parallel', but got the num_heads is {} and "
                    "parallel_config.model_parallel is {}.".format(
                        num_heads, parallel_config.model_parallel
                    )
                )
            if hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'hidden_size' must be divisibled by "
                    "the 'parallel_config.model_parallel', but got the hidden_size is {} and parallel_config."
                    " model_parallel is {}.".format(
                        hidden_size, parallel_config.model_parallel
                    )
                )
            if ffn_hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'ffn_hidden_size' must be divisibled "
                    "by the 'parallel_config.model_parallel', but got the ffn_hidden_size is {} "
                    "and parallel_config. model_parallel is {}.".format(
                        ffn_hidden_size, parallel_config.model_parallel
                    )
                )
            _check_moe_config(moe_config, parallel_config)
            self.use_moe = moe_config.expert_num > 1
            self.use_past = use_past
            self.seq_length = seq_length
            self.hidden_size = hidden_size
            self.batch_size = batch_size
            self.layernorm1 = _LayerNorm((hidden_size,)).to_float(
                layernorm_compute_type
            )
            self.layernorm1.shard(((parallel_config.data_parallel, 1),))
            self.layernorm2 = _LayerNorm((hidden_size,)).to_float(
                layernorm_compute_type
            )
            self.layernorm2.shard(((parallel_config.data_parallel, 1),))
            parallel_config_args = (
                parallel_config.dpmp if self.use_moe else parallel_config
            )
            self.attention = MultiHeadAttention(
                batch_size=batch_size,
                src_seq_length=seq_length,
                tgt_seq_length=seq_length,
                hidden_size=hidden_size,
                num_heads=num_heads,
                hidden_dropout_rate=hidden_dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                softmax_compute_type=softmax_compute_type,
                param_init_type=param_init_type,
                use_past=use_past,
                parallel_config=parallel_config_args,
            )

            # For ABINet, the following paragraph is deleted
            # if self.use_moe:
            #     self.output = MoE(hidden_size=hidden_size,
            #                       dropout_rate=hidden_dropout_rate,
            #                       ffn_hidden_size=ffn_hidden_size,
            #                       param_init_type=param_init_type,
            #                       hidden_act=hidden_act,
            #                       moe_config=moe_config,
            #                       parallel_config=parallel_config)
            # else:
            # Feed Forward Network, FFN
            self.output = FeedForward(
                hidden_size=hidden_size,
                dropout_rate=hidden_dropout_rate,
                ffn_hidden_size=ffn_hidden_size,
                param_init_type=param_init_type,
                hidden_act=hidden_act,
                parallel_config=parallel_config,
            )
            self.post_layernorm_residual = post_layernorm_residual
            self.add = P.Add().shard(
                ((parallel_config.data_parallel, 1), (parallel_config.data_parallel, 1))
            )
            self.add_3d = P.Add().shard(
                (
                    (parallel_config.data_parallel, 1, 1),
                    (parallel_config.data_parallel, 1, 1),
                )
            )
            self.dtype = mstype.float16
            self.key_past = None
            self.value_past = None

            # For ABINet, the following paragraph is deleted
            # if self.use_past:
            #     # operator used for state reuse
            #     self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
            #     self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
            #     self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
            #     size_per_head = int(hidden_size / num_heads)
            #     self.key_shape = (batch_size, num_heads, size_per_head, seq_length)
            #     self.value_shape = (batch_size, num_heads, seq_length, size_per_head)
            #     # parameters saving key and value states
            #     self.key_past = Parameter(Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past")
            #     self.value_past = Parameter(Tensor(np.zeros(shape=self.value_shape), self.dtype), name="value_past")
            #     self.tile = P.Tile().shard(((1, 1),))
            #     self.mul = P.Mul().shard(((1, 1, 1, 1), (1,)))
            #     self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        else:
            raise RuntimeError(
                f"The {self.cls_name} only support sharding propagation or "
                f"semi-auto parallel mode now."
            )

    def construct(self, x, input_mask, init_reset=True, batch_valid_length=None):
        self._check_input(x, input_mask, init_reset, batch_valid_length)
        x_shape = F.shape(x)
        x = F.reshape(x, (-1, x_shape[-1]))

        # For ABINet, the following paragraph needs to be revised
        # input_x = self.layernorm1(x)
        input_x = F.cast(x, self.dtype)

        # indicate whether reset saved states
        # key_reset = None
        # value_reset = None

        # if self.use_past:
        #     # reset states, init_reset True for reuse and False for reset
        #     key_reset = self.assign(self.key_past, self.mul(self.key_past, F.cast(init_reset, self.dtype)))
        #     value_reset = self.assign(self.value_past, self.mul(self.value_past, F.cast(init_reset, self.dtype)))
        #     # add dependency for desired execution order
        #     input_x = F.depend(input_x, key_reset)
        #     input_x = F.depend(input_x, value_reset)

        attention, layer_present = self.attention(
            input_x,
            input_x,
            input_x,
            input_mask,
            self.key_past,
            self.value_past,
            batch_valid_length,
        )
        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        # if self.post_layernorm_residual:
        #     x = self.add(input_x, attention)
        # # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        # else:
        #     x = self.add(x, attention)
        x = self.add(x, attention)
        x = self.layernorm1(x)

        # output_x = self.layernorm2(x)
        output_x = F.cast(x, self.dtype)
        # if self.use_moe:
        #     mlp_logit, aux_loss = self.output(output_x)

        mlp_logit = self.output(output_x)

        value_update = None
        key_update = None
        # if self.use_past:
        #     # current key and value
        #     key_present, value_present = layer_present
        #     # update key and value calculated this step
        #     key_update = self.assign(self.key_past, key_present)
        #     value_update = self.assign(self.value_past, value_present)
        #     # add dependency for desired execution order
        #     key_update = F.depend(key_update, key_reset)
        #     value_update = F.depend(value_update, value_reset)

        # add dependency for desired execution order
        mlp_logit = F.depend(mlp_logit, value_update)
        mlp_logit = F.depend(mlp_logit, key_update)
        output = 1.0  # mindspore graph need assignment
        # if shape is 3d, we reshape the inputs of the add
        if len(x_shape) == 3:
            output_x = P.Reshape()(output_x, x_shape)
            mlp_logit = P.Reshape()(mlp_logit, x_shape)
            x = P.Reshape()(x, x_shape)

            # if self.post_layernorm_residual:
            #     output = self.add_3d(output_x, mlp_logit)
            # else:
            #     output = self.add_3d(x, mlp_logit)
            output = self.add_3d(x, mlp_logit)
            output = self.layernorm2(output)

        else:
            # if self.post_layernorm_residual:
            #     output = self.add(output_x, mlp_logit)
            # else:
            #     output = self.add(x, mlp_logit)
            output = F.reshape(output, x_shape)
            output = self.add_3d(x, mlp_logit)
            output = self.layernorm2(output)
        # if self.use_moe:
        #     return output, layer_present, aux_loss
        return output

    def _check_input(self, x, input_mask, init_reset, batch_valid_length):
        r"""Check inputs"""
        if not self.use_past or (self.use_past and self.is_first_iteration):
            _check_shape_equal(
                F.shape(x),
                "x",
                self.cls_name,
                [
                    [self.batch_size, self.seq_length, self.hidden_size],
                    [self.batch_size * self.seq_length, self.hidden_size],
                ],
            )
            _check_shape_equal(
                F.shape(input_mask),
                "input_mask",
                self.cls_name,
                [self.batch_size, self.seq_length, self.seq_length],
            )
        else:
            _check_shape_equal(
                F.shape(x), "x", self.cls_name, [self.batch_size, 1, self.hidden_size]
            )
            _check_shape_equal(
                F.shape(input_mask),
                "input_mask",
                self.cls_name,
                [self.batch_size, 1, self.seq_length],
            )
        _check_input_dtype(
            F.dtype(x), "x", [mstype.float32, mstype.float16], self.cls_name
        )
        _check_input_dtype(
            F.dtype(input_mask),
            "input_mask",
            [mstype.float32, mstype.float16],
            self.cls_name,
        )

        init_reset_is_tensor = isinstance(init_reset, Tensor)
        init_reset_is_default = init_reset is True
        batch_valid_length_is_tensor = isinstance(batch_valid_length, Tensor)
        batch_is_default = batch_valid_length is None
        _check_past_none_input_none(
            self.use_past,
            "init_reset",
            self.cls_name,
            True,
            init_reset_is_tensor,
            init_reset_is_default,
        )
        _check_past_none_input_none(
            self.use_past,
            "batch_valid_length",
            self.cls_name,
            None,
            batch_valid_length_is_tensor,
            batch_is_default,
        )

        if self.use_past:
            _check_shape_equal(F.shape(init_reset), "init_reset", self.cls_name, [1])
            _check_input_dtype(
                F.dtype(init_reset), "init_reset", [mstype.bool_], self.cls_name
            )
            _check_shape_equal(
                F.shape(batch_valid_length),
                "batch_valid_length",
                self.cls_name,
                [self.batch_size],
            )
            _check_input_dtype(
                F.dtype(batch_valid_length),
                "batch_valid_length",
                [mstype.int32],
                self.cls_name,
            )
        return True


class TransformerDecoderLayer(Cell):

    @_LogActionOnce(
        logger=logger,
        key="TransformerDecoderLayer",
        no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,),
    )
    @_args_type_validator_check(
        batch_size=Validator.check_positive_int,
        hidden_size=Validator.check_positive_int,
        num_heads=Validator.check_positive_int,
        ffn_hidden_size=Validator.check_positive_int,
        src_seq_length=Validator.check_positive_int,
        tgt_seq_length=Validator.check_positive_int,
        attention_dropout_rate=Validator.check_non_negative_float,
        hidden_dropout_rate=Validator.check_non_negative_float,
        hidden_act=_valid_type_checks([str], "TransformerDecoderLayer"),
        post_layernorm_residual=Validator.check_bool,
        layernorm_compute_type=_valid_value_checks(
            [mstype.float32, mstype.float16], "TransformerDecoderLayer"
        ),
        softmax_compute_type=_valid_value_checks(
            [mstype.float32, mstype.float16], "TransformerDecoderLayer"
        ),
        param_init_type=_valid_value_checks(
            [mstype.float32, mstype.float16], "TransformerDecoderLayer"
        ),
        parallel_config=_valid_type_checks(
            [OpParallelConfig, MoEParallelConfig], "TransformerDecoderLayer"
        ),
        use_past=Validator.check_bool,
    )
    def __init__(
        self,
        hidden_size,
        ffn_hidden_size,
        num_heads,
        batch_size,
        src_seq_length,
        tgt_seq_length,
        attention_dropout_rate=0.1,
        hidden_dropout_rate=0.1,
        post_layernorm_residual=False,
        use_past=False,
        layernorm_compute_type=mstype.float32,
        softmax_compute_type=mstype.float32,
        param_init_type=mstype.float32,
        hidden_act="gelu",
        moe_config=default_moe_config,
        parallel_config=default_dpmp_config,
    ):
        super(TransformerDecoderLayer, self).__init__()
        if (
            _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,)
            and _is_sharding_propagation()
        ):
            _check_config(parallel_config)
            if num_heads % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerDecoderLayer', the class variable 'num_heads' must be divisibled by "
                    "'parallel_config.model_parallel', but got the num_heads is {} and "
                    "parallel_config.model_parallel is {}.".format(
                        num_heads, parallel_config.model_parallel
                    )
                )
            if hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerDecoderLayer', the class variable 'hidden_size' must be divisibled by "
                    "'parallel_config.model_parallel', but got the hidden_size is {} and "
                    "parallel_config.model_parallel is {}.".format(
                        hidden_size, parallel_config.model_parallel
                    )
                )
            if ffn_hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerDecoderLayer', the class variable 'ffn_hidden_size' must be "
                    "divisibled by 'parallel_config.model_parallel', but got the ffn_hidden_size is {} "
                    "and parallel_config.model_parallel is {}.".format(
                        ffn_hidden_size, parallel_config.model_parallel
                    )
                )
            _check_moe_config(moe_config, parallel_config)
            self.use_moe = moe_config.expert_num > 1
            if use_past:
                raise ValueError(f"The {self.cls_name} does not support use_past=True.")
            self.batch_size = batch_size
            self.use_past = use_past
            self.softmax_compute_type = softmax_compute_type

            self.src_seq_length = src_seq_length
            self.tgt_seq_length = tgt_seq_length
            self.use_past = use_past
            self.hidden_size = hidden_size

            # For ABINet, the following paragraph needs to be revised
            # self.layernorm1 = _LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
            self.layernorm2 = _LayerNorm((hidden_size,)).to_float(
                layernorm_compute_type
            )
            parallel_config_args = (
                parallel_config.dpmp if self.use_moe else parallel_config
            )
            # self.attention = MultiHeadAttention(hidden_size=hidden_size,
            #                                     num_heads=num_heads,
            #                                     batch_size=batch_size,
            #                                     src_seq_length=tgt_seq_length,
            #                                     tgt_seq_length=tgt_seq_length,
            #                                     hidden_dropout_rate=hidden_dropout_rate,
            #                                     attention_dropout_rate=attention_dropout_rate,
            #                                     use_past=use_past,
            #                                     softmax_compute_type=softmax_compute_type,
            #                                     param_init_type=param_init_type,
            #                                     parallel_config=parallel_config_args)

            # Cross attention with the output of encoder as memory tensor
            self.cross_attention = MultiHeadAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                batch_size=batch_size,
                src_seq_length=tgt_seq_length,
                tgt_seq_length=src_seq_length,
                hidden_dropout_rate=hidden_dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                softmax_compute_type=softmax_compute_type,
                use_past=use_past,
                param_init_type=param_init_type,
                parallel_config=parallel_config_args,
            )
            self.cross_attention_layernorm = _LayerNorm((hidden_size,)).to_float(
                layernorm_compute_type
            )

            if self.use_moe:
                self.output = MoE(
                    hidden_size=hidden_size,
                    dropout_rate=hidden_dropout_rate,
                    ffn_hidden_size=ffn_hidden_size,
                    param_init_type=param_init_type,
                    hidden_act=hidden_act,
                    moe_config=moe_config,
                    parallel_config=parallel_config,
                )
            else:
                # Feed Forward Network, FFN
                self.output = FeedForward(
                    hidden_size=hidden_size,
                    dropout_rate=hidden_dropout_rate,
                    ffn_hidden_size=ffn_hidden_size,
                    hidden_act=hidden_act,
                    param_init_type=param_init_type,
                    parallel_config=parallel_config,
                )
            self.post_layernorm_residual = post_layernorm_residual
            self.add = P.Add()
            self.add_3d = P.Add()
            self.dtype = mstype.float16
            self.key_past = None
            self.value_past = None
            if self.use_past:
                # operator used for state reuse
                self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
                self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
                self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
                size_per_head = int(hidden_size / num_heads)
                self.key_shape = (batch_size, num_heads, size_per_head, tgt_seq_length)
                self.value_shape = (
                    batch_size,
                    num_heads,
                    tgt_seq_length,
                    size_per_head,
                )
                # parameters saving key and value states
                self.key_past = Parameter(
                    Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past"
                )
                self.value_past = Parameter(
                    Tensor(np.zeros(shape=self.value_shape), self.dtype),
                    name="value_past",
                )
                self.tile = P.Tile().shard(((1, 1),))
                self.mul = P.Mul().shard(((1, 1, 1, 1), (1,)))
                self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        elif _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            _check_config(parallel_config)
            if num_heads % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerDecoderLayer', the class variable 'num_heads' must be divisibled by "
                    "'parallel_config.model_parallel', but got the num_heads is {} and "
                    "parallel_config.model_parallel is {}.".format(
                        num_heads, parallel_config.model_parallel
                    )
                )
            if hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerDecoderLayer', the class variable 'hidden_size' must be divisibled by "
                    "'parallel_config.model_parallel', but got the hidden_size is {} and "
                    "parallel_config.model_parallel is {}.".format(
                        hidden_size, parallel_config.model_parallel
                    )
                )
            if ffn_hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerDecoderLayer', the class variable 'ffn_hidden_size' must be "
                    "divisibled by 'parallel_config.model_parallel', but got the ffn_hidden_size is {} "
                    "and parallel_config.model_parallel is {}.".format(
                        ffn_hidden_size, parallel_config.model_parallel
                    )
                )
            _check_moe_config(moe_config, parallel_config)
            self.use_moe = moe_config.expert_num > 1
            if use_past:
                raise ValueError(f"The {self.cls_name} does not support use_past=True.")
            self.batch_size = batch_size
            self.use_past = use_past
            self.softmax_compute_type = softmax_compute_type

            self.src_seq_length = src_seq_length
            self.tgt_seq_length = tgt_seq_length
            self.use_past = use_past
            self.hidden_size = hidden_size

            # self.layernorm1 = _LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
            # self.layernorm1.shard(((parallel_config.data_parallel, 1),))
            self.layernorm2 = _LayerNorm((hidden_size,)).to_float(
                layernorm_compute_type
            )
            self.layernorm2.shard(((parallel_config.data_parallel, 1),))
            parallel_config_args = (
                parallel_config.dpmp if self.use_moe else parallel_config
            )
            # self.attention = MultiHeadAttention(hidden_size=hidden_size,
            #                                     num_heads=num_heads,
            #                                     batch_size=batch_size,
            #                                     src_seq_length=tgt_seq_length,
            #                                     tgt_seq_length=tgt_seq_length,
            #                                     hidden_dropout_rate=hidden_dropout_rate,
            #                                     attention_dropout_rate=attention_dropout_rate,
            #                                     use_past=use_past,
            #                                     softmax_compute_type=softmax_compute_type,
            #                                     param_init_type=param_init_type,
            #                                     parallel_config=parallel_config_args)

            # Cross attention with the output of encoder as memory tensor
            self.cross_attention = MultiHeadAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                batch_size=batch_size,
                src_seq_length=tgt_seq_length,
                tgt_seq_length=src_seq_length,
                hidden_dropout_rate=hidden_dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                softmax_compute_type=softmax_compute_type,
                use_past=use_past,
                param_init_type=param_init_type,
                parallel_config=parallel_config_args,
            )
            self.cross_attention_layernorm = _LayerNorm((hidden_size,)).to_float(
                layernorm_compute_type
            )
            self.cross_attention_layernorm.shard(((parallel_config.data_parallel, 1),))

            if self.use_moe:
                self.output = MoE(
                    hidden_size=hidden_size,
                    dropout_rate=hidden_dropout_rate,
                    ffn_hidden_size=ffn_hidden_size,
                    param_init_type=param_init_type,
                    hidden_act=hidden_act,
                    moe_config=moe_config,
                    parallel_config=parallel_config,
                )
            else:
                # Feed Forward Network, FFN
                self.output = FeedForward(
                    hidden_size=hidden_size,
                    dropout_rate=hidden_dropout_rate,
                    ffn_hidden_size=ffn_hidden_size,
                    hidden_act=hidden_act,
                    param_init_type=param_init_type,
                    parallel_config=parallel_config,
                )
            self.post_layernorm_residual = post_layernorm_residual
            self.add = P.Add().shard(
                ((parallel_config.data_parallel, 1), (parallel_config.data_parallel, 1))
            )
            self.add_3d = P.Add().shard(
                (
                    (parallel_config.data_parallel, 1, 1),
                    (parallel_config.data_parallel, 1, 1),
                )
            )
            self.dtype = mstype.float16
            self.key_past = None
            self.value_past = None
            if self.use_past:
                # operator used for state reuse
                self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
                self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
                self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
                size_per_head = int(hidden_size / num_heads)
                self.key_shape = (batch_size, num_heads, size_per_head, tgt_seq_length)
                self.value_shape = (
                    batch_size,
                    num_heads,
                    tgt_seq_length,
                    size_per_head,
                )
                # parameters saving key and value states
                self.key_past = Parameter(
                    Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past"
                )
                self.value_past = Parameter(
                    Tensor(np.zeros(shape=self.value_shape), self.dtype),
                    name="value_past",
                )
                self.tile = P.Tile().shard(((1, 1),))
                self.mul = P.Mul().shard(((1, 1, 1, 1), (1,)))
                self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        else:
            raise RuntimeError(
                f"The {self.cls_name} only support sharding propagation or "
                f"semi-auto parallel mode now."
            )

    def construct(
        self,
        hidden_stats,
        decoder_mask,
        encoder_output=None,
        memory_mask=None,
        init_reset=True,
        batch_valid_length=None,
    ):
        self._check_input(
            hidden_stats,
            decoder_mask,
            encoder_output,
            memory_mask,
            init_reset,
            batch_valid_length,
        )
        # the returned shape is [bs, seq_length, embedding_size] or [bs * seq_length, embedding_size]
        hidden_shape = F.shape(hidden_stats)
        hidden_stats = F.reshape(hidden_stats, (-1, hidden_shape[-1]))
        # input_x = self.layernorm1(hidden_stats)
        # input_x = F.cast(input_x, self.dtype)

        # # indicate whether reset saved states
        # key_reset = None
        # value_reset = None
        # if self.use_past:
        #     # reset states, init_reset True for reuse and False for reset
        #     key_reset = self.assign(self.key_past, self.mul(self.key_past, F.cast(init_reset, self.dtype)))
        #     value_reset = self.assign(self.value_past, self.mul(self.value_past, F.cast(init_reset, self.dtype)))
        #     # add dependency for desired execution order
        #     input_x = F.depend(input_x, key_reset)
        #     input_x = F.depend(input_x, value_reset)

        # attention, layer_present = self.attention(input_x, input_x, input_x, decoder_mask, self.key_past,
        #                                           self.value_past, batch_valid_length)
        # # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        # if self.post_layernorm_residual:
        #     x = self.add(input_x, attention)
        # # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        # else:
        #     x = self.add(hidden_stats, attention)
        x = hidden_stats  # 为了适应pytorch没有调用self.attn
        middle_output = None
        if encoder_output is not None:
            # middle_output = self.cross_attention_layernorm(x)
            middle_output = F.cast(x, self.dtype)
            encoder_output = F.cast(encoder_output, self.dtype)
            cross_attn_output, cross_layer_present = self.cross_attention(
                middle_output,
                encoder_output,
                encoder_output,
                memory_mask,
                self.key_past,
                self.value_past,
                batch_valid_length,
            )
            # layer_present += cross_layer_present
            # if self.post_layernorm_residual:
            #     x = self.add(middle_output, cross_attn_output)
            # else:
            #     x = self.add(x, cross_attn_output)
            x = self.add(x, cross_attn_output)
            x = self.cross_attention_layernorm(x)

        # output_x = self.layernorm2(x)
        output_x = F.cast(x, self.dtype)
        aux_loss = None
        if self.use_moe:
            mlp_logit, aux_loss = self.output(output_x)
        else:
            mlp_logit = self.output(output_x)

        value_update = None
        key_update = None
        # if self.use_past:
        #     # current key and value
        #     key_present, value_present = layer_present
        #     # update key and value calculated this step
        #     key_update = self.assign(self.key_past, key_present)
        #     value_update = self.assign(self.value_past, value_present)
        #     # add dependency for desired execution order
        #     key_update = F.depend(key_update, key_reset)
        #     value_update = F.depend(value_update, value_reset)

        # add dependency for desired execution order
        mlp_logit = F.depend(mlp_logit, value_update)
        mlp_logit = F.depend(mlp_logit, key_update)

        # if shape is 3d, we reshape the inputs of the add
        if len(hidden_shape) == 3:
            output_x = P.Reshape()(output_x, hidden_shape)
            mlp_logit = P.Reshape()(mlp_logit, hidden_shape)
            x = P.Reshape()(x, hidden_shape)

            # if self.post_layernorm_residual:
            #     output = self.add_3d(output_x, mlp_logit)
            # else:
            #     output = self.add_3d(x, mlp_logit)
            output = self.add_3d(x, mlp_logit)
            output = self.layernorm2(output)

        else:
            # if self.post_layernorm_residual:
            #     output = self.add(output_x, mlp_logit)
            # else:
            #     output = self.add(x, mlp_logit)
            # output = F.reshape(output, hidden_shape)
            output = self.add(x, mlp_logit)
            output = self.layernorm2(output)

        # if self.use_moe:
        #     return output, layer_present, aux_loss
        return output

    def _check_input(
        self,
        hidden_states,
        attention_mask,
        encoder_output,
        memory_mask,
        init_reset,
        batch_valid_length,
    ):
        r"""Check inputs"""
        if not self.use_past or (self.use_past and self.is_first_iteration):
            _check_shape_equal(
                F.shape(hidden_states),
                "hidden_states",
                self.cls_name,
                [
                    [self.batch_size, self.tgt_seq_length, self.hidden_size],
                    [self.batch_size * self.tgt_seq_length, self.hidden_size],
                ],
            )
            _check_shape_equal(
                F.shape(attention_mask),
                "attention_mask",
                self.cls_name,
                [self.batch_size, self.tgt_seq_length, self.tgt_seq_length],
            )

        else:
            _check_shape_equal(
                F.shape(hidden_states),
                "hidden_states",
                self.cls_name,
                [self.batch_size, 1, self.hidden_size],
            )
            _check_shape_equal(
                F.shape(attention_mask),
                "attention_mask",
                self.cls_name,
                [self.batch_size, 1, self.tgt_seq_length],
            )
        _check_input_dtype(
            F.dtype(hidden_states),
            "hidden_states",
            [mstype.float32, mstype.float16],
            self.cls_name,
        )
        _check_input_dtype(
            F.dtype(attention_mask),
            "attention_mask",
            [mstype.float32, mstype.float16],
            self.cls_name,
        )
        if encoder_output is not None:
            _check_shape_equal(
                F.shape(encoder_output),
                "encoder_output",
                self.cls_name,
                [
                    [self.batch_size, self.src_seq_length, self.hidden_size],
                    [self.batch_size * self.src_seq_length, self.hidden_size],
                ],
            )
            _check_input_dtype(
                F.dtype(encoder_output),
                "encoder_output",
                [mstype.float32, mstype.float16],
                self.cls_name,
            )
        if memory_mask is not None:
            _check_shape_equal(
                F.shape(memory_mask),
                "memory_mask",
                self.cls_name,
                [self.batch_size, self.tgt_seq_length, self.src_seq_length],
            )
            _check_input_dtype(
                F.dtype(memory_mask),
                "memory_mask",
                [mstype.float32, mstype.float16],
                self.cls_name,
            )

        init_reset_is_tensor = isinstance(init_reset, Tensor)
        init_reset_is_default = init_reset is True
        batch_valid_length_is_tensor = isinstance(batch_valid_length, Tensor)
        batch_is_default = batch_valid_length is None
        _check_past_none_input_none(
            self.use_past,
            "init_reset",
            self.cls_name,
            True,
            init_reset_is_tensor,
            init_reset_is_default,
        )
        _check_past_none_input_none(
            self.use_past,
            "batch_valid_length",
            self.cls_name,
            None,
            batch_valid_length_is_tensor,
            batch_is_default,
        )

        if self.use_past:
            _check_shape_equal(F.shape(init_reset), "init_reset", self.cls_name, [1])
            _check_input_dtype(
                F.dtype(init_reset), "init_reset", [mstype.bool_], self.cls_name
            )
            _check_shape_equal(
                F.shape(batch_valid_length),
                "batch_valid_length",
                self.cls_name,
                [self.batch_size],
            )
            _check_input_dtype(
                F.dtype(batch_valid_length),
                "batch_valid_length",
                [mstype.int32],
                self.cls_name,
            )
        return True


class TransformerEncoder(Cell):

    @_LogActionOnce(
        logger=logger,
        key="TransformerEncoder",
        no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,),
    )
    @_args_type_validator_check(
        batch_size=Validator.check_positive_int,
        hidden_size=Validator.check_positive_int,
        num_heads=Validator.check_positive_int,
        ffn_hidden_size=Validator.check_positive_int,
        seq_length=Validator.check_positive_int,
        num_layers=Validator.check_positive_int,
        offset=Validator.check_non_negative_int,
        attention_dropout_rate=Validator.check_non_negative_float,
        hidden_dropout_rate=Validator.check_non_negative_float,
        hidden_act=_valid_type_checks([str], "TransformerEncoder"),
        post_layernorm_residual=Validator.check_bool,
        layernorm_compute_type=_valid_value_checks(
            [mstype.float32, mstype.float16], "TransformerEncoder"
        ),
        softmax_compute_type=_valid_value_checks(
            [mstype.float32, mstype.float16], "TransformerEncoder"
        ),
        param_init_type=_valid_value_checks(
            [mstype.float32, mstype.float16], "TransformerEncoder"
        ),
        parallel_config=_valid_type_checks(
            [TransformerOpParallelConfig], "TransformerEncoder"
        ),
        use_past=Validator.check_bool,
    )
    def __init__(
        self,
        batch_size,
        num_layers,
        hidden_size,
        ffn_hidden_size,
        seq_length,
        num_heads,
        attention_dropout_rate=0.1,
        hidden_dropout_rate=0.1,
        hidden_act="gelu",
        post_layernorm_residual=False,
        layernorm_compute_type=mstype.float32,
        softmax_compute_type=mstype.float32,
        param_init_type=mstype.float32,
        lambda_func=None,
        offset=0,
        use_past=False,
        moe_config=default_moe_config,
        parallel_config=default_transformer_config,
    ):
        super(TransformerEncoder, self).__init__()
        if (
            _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,)
            and _is_sharding_propagation()
        ):
            _check_config(parallel_config)
            _check_moe_config(moe_config, parallel_config)
            self.use_moe = moe_config.expert_num > 1
            self.add = P.Add()
            self.aux_loss = Tensor(0.0, mstype.float32)
            self.num_layers = num_layers
            self.blocks = nn.CellList()
            parallel_config_args = (
                parallel_config.moe_parallel_config
                if self.use_moe
                else parallel_config.dp_mp_config
            )
            for i in range(num_layers):
                block = TransformerEncoderLayer(
                    hidden_size=hidden_size,
                    batch_size=batch_size,
                    ffn_hidden_size=ffn_hidden_size,
                    seq_length=seq_length,
                    attention_dropout_rate=attention_dropout_rate,
                    hidden_dropout_rate=hidden_dropout_rate,
                    layernorm_compute_type=layernorm_compute_type,
                    softmax_compute_type=softmax_compute_type,
                    num_heads=num_heads,
                    hidden_act=hidden_act,
                    post_layernorm_residual=post_layernorm_residual,
                    param_init_type=param_init_type,
                    use_past=use_past,
                    moe_config=moe_config,
                    parallel_config=parallel_config_args,
                )
                # If the user doesn't pass the fusion function, use the default one
                if not lambda_func:
                    lambda_func = _get_lambda_func()

                lambda_func(
                    block,
                    layer_id=i,
                    layers=num_layers,
                    offset=offset,
                    parallel_config=parallel_config,
                )
                self.blocks.append(block)
        elif _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            _check_config(parallel_config)
            _check_moe_config(moe_config, parallel_config)
            self.use_moe = moe_config.expert_num > 1
            self.add = P.Add().shard(((), ()))
            self.aux_loss = Tensor(0.0, mstype.float32)
            logger.warning(
                "For parallel mode, sharding propagation is recommended, you can use it by setting "
                "'set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, "
                'search_mode="sharding_propagation")\' and '
                "'set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)'"
            )
            self.num_layers = num_layers
            self.blocks = nn.CellList()
            parallel_config_args = (
                parallel_config.moe_parallel_config
                if self.use_moe
                else parallel_config.dp_mp_config
            )
            for i in range(num_layers):
                block = TransformerEncoderLayer(
                    hidden_size=hidden_size,
                    batch_size=batch_size,
                    ffn_hidden_size=ffn_hidden_size,
                    seq_length=seq_length,
                    attention_dropout_rate=attention_dropout_rate,
                    hidden_dropout_rate=hidden_dropout_rate,
                    layernorm_compute_type=layernorm_compute_type,
                    softmax_compute_type=softmax_compute_type,
                    num_heads=num_heads,
                    hidden_act=hidden_act,
                    post_layernorm_residual=post_layernorm_residual,
                    param_init_type=param_init_type,
                    use_past=use_past,
                    moe_config=moe_config,
                    parallel_config=parallel_config_args,
                )
                # If the user doesn't pass the fusion function, use the default one
                if not lambda_func:
                    lambda_func = _get_lambda_func()

                lambda_func(
                    block,
                    layer_id=i,
                    layers=num_layers,
                    offset=offset,
                    parallel_config=parallel_config,
                )
                self.blocks.append(block)
        else:
            raise RuntimeError(
                f"The {self.cls_name} only support sharding propagation or "
                f"semi-auto parallel mode now."
            )

    def construct(
        self, hidden_states, attention_mask, init_reset=True, batch_valid_length=None
    ):

        # if self.use_moe:
        #     accum_loss = self.aux_loss
        #     for i in range(self.num_layers):
        #         hidden_states, present, aux_loss = self.blocks[i](hidden_states,
        #                                                           attention_mask,
        #                                                           init_reset,
        #                                                           batch_valid_length)
        #         present_layer = present_layer + (present,)
        #         accum_loss = self.add(accum_loss, aux_loss)
        #     return hidden_states, present_layer, accum_loss

        for i in range(self.num_layers):
            hidden_states = self.blocks[i](
                hidden_states, attention_mask, init_reset, batch_valid_length
            )
            # present_layer = present_layer + (present,)

        return hidden_states


class TransformerDecoder(Cell):

    @_LogActionOnce(
        logger=logger,
        key="TransformerDecoder",
        no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,),
    )
    @_args_type_validator_check(
        batch_size=Validator.check_positive_int,
        hidden_size=Validator.check_positive_int,
        num_heads=Validator.check_positive_int,
        ffn_hidden_size=Validator.check_positive_int,
        src_seq_length=Validator.check_positive_int,
        num_layers=Validator.check_positive_int,
        tgt_seq_length=Validator.check_positive_int,
        offset=Validator.check_non_negative_int,
        attention_dropout_rate=Validator.check_non_negative_float,
        hidden_dropout_rate=Validator.check_non_negative_float,
        hidden_act=_valid_type_checks([str], "TransformerDecoder"),
        post_layernorm_residual=Validator.check_bool,
        layernorm_compute_type=_valid_value_checks(
            [mstype.float32, mstype.float16], "TransformerDecoder"
        ),
        softmax_compute_type=_valid_value_checks(
            [mstype.float32, mstype.float16], "TransformerDecoder"
        ),
        param_init_type=_valid_value_checks(
            [mstype.float32, mstype.float16], "TransformerDecoder"
        ),
        parallel_config=_valid_type_checks(
            [TransformerOpParallelConfig], "TransformerDecoder"
        ),
        use_past=Validator.check_bool,
    )
    def __init__(
        self,
        num_layers,
        batch_size,
        hidden_size,
        ffn_hidden_size,
        src_seq_length,
        tgt_seq_length,
        num_heads,
        attention_dropout_rate=0.1,
        hidden_dropout_rate=0.1,
        post_layernorm_residual=False,
        layernorm_compute_type=mstype.float32,
        softmax_compute_type=mstype.float32,
        param_init_type=mstype.float32,
        hidden_act="gelu",
        lambda_func=None,
        use_past=False,
        offset=0,
        moe_config=default_moe_config,
        parallel_config=default_transformer_config,
    ):
        super(TransformerDecoder, self).__init__()
        if (
            _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,)
            and _is_sharding_propagation()
        ):
            _check_config(parallel_config)

            self.add = P.Add()
            self.aux_loss = Tensor(0.0, mstype.float32)
            self.num_layers = num_layers
            self.blocks = nn.CellList()
            _check_moe_config(moe_config, parallel_config)
            self.use_moe = moe_config.expert_num > 1
            parallel_config_args = (
                parallel_config.moe_parallel_config
                if self.use_moe
                else parallel_config.dp_mp_config
            )
            for i in range(num_layers):
                block = TransformerDecoderLayer(
                    hidden_size=hidden_size,
                    batch_size=batch_size,
                    ffn_hidden_size=ffn_hidden_size,
                    src_seq_length=src_seq_length,
                    tgt_seq_length=tgt_seq_length,
                    attention_dropout_rate=attention_dropout_rate,
                    hidden_dropout_rate=hidden_dropout_rate,
                    num_heads=num_heads,
                    layernorm_compute_type=layernorm_compute_type,
                    softmax_compute_type=softmax_compute_type,
                    hidden_act=hidden_act,
                    use_past=use_past,
                    param_init_type=param_init_type,
                    post_layernorm_residual=post_layernorm_residual,
                    moe_config=moe_config,
                    parallel_config=parallel_config_args,
                )
                # If the user doesn't pass the fusion function, use the default one
                if not lambda_func:
                    lambda_func = _get_lambda_func()

                lambda_func(
                    block,
                    layer_id=i,
                    layers=num_layers,
                    offset=offset,
                    parallel_config=parallel_config,
                )

                self.blocks.append(block)
        elif _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            _check_config(parallel_config)

            self.add = P.Add().shard(((), ()))
            self.aux_loss = Tensor(0.0, mstype.float32)
            logger.warning(
                "For parallel mode, sharding propagation is recommended, you can use it by setting "
                "'set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, "
                'search_mode="sharding_propagation")\' and '
                "'set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)'"
            )
            self.num_layers = num_layers
            self.blocks = nn.CellList()
            _check_moe_config(moe_config, parallel_config)
            self.use_moe = moe_config.expert_num > 1
            parallel_config_args = (
                parallel_config.moe_parallel_config
                if self.use_moe
                else parallel_config.dp_mp_config
            )
            for i in range(num_layers):
                block = TransformerDecoderLayer(
                    hidden_size=hidden_size,
                    batch_size=batch_size,
                    ffn_hidden_size=ffn_hidden_size,
                    src_seq_length=src_seq_length,
                    tgt_seq_length=tgt_seq_length,
                    attention_dropout_rate=attention_dropout_rate,
                    hidden_dropout_rate=hidden_dropout_rate,
                    num_heads=num_heads,
                    layernorm_compute_type=layernorm_compute_type,
                    softmax_compute_type=softmax_compute_type,
                    hidden_act=hidden_act,
                    use_past=use_past,
                    param_init_type=param_init_type,
                    post_layernorm_residual=post_layernorm_residual,
                    moe_config=moe_config,
                    parallel_config=parallel_config_args,
                )
                # If the user doesn't pass the fusion function, use the default one
                if not lambda_func:
                    lambda_func = _get_lambda_func()

                lambda_func(
                    block,
                    layer_id=i,
                    layers=num_layers,
                    offset=offset,
                    parallel_config=parallel_config,
                )

                self.blocks.append(block)
        else:
            raise RuntimeError(
                f"The {self.cls_name} only support sharding propagation or "
                f"semi-auto parallel mode now."
            )

    def construct(
        self,
        hidden_states,
        attention_mask,
        encoder_output=None,
        memory_mask=None,
        init_reset=True,
        batch_valid_length=None,
    ):

        # if self.use_moe:
        #     accum_loss = self.aux_loss
        #     for i in range(self.num_layers):
        #         hidden_states, present, aux_loss = self.blocks[i](hidden_states,
        #                                                           attention_mask,
        #                                                           encoder_output,
        #                                                           memory_mask,
        #                                                           init_reset,
        #                                                           batch_valid_length)
        #         present_layer = present_layer + (present,)
        #         accum_loss = self.add(accum_loss, aux_loss)
        #     return hidden_states, present_layer, accum_loss

        # Loop through each self-attention layer
        for i in range(self.num_layers):
            hidden_states = self.blocks[i](
                hidden_states,
                attention_mask,
                encoder_output,
                memory_mask,
                init_reset,
                batch_valid_length,
            )
            # present_layer = present_layer + (present,)

        return hidden_states
