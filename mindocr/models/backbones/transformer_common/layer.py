import math

import numpy as np

from mindspore import Parameter, Tensor, nn, ops
from mindspore.common import dtype as mstype
from mindspore.common.initializer import Constant, initializer

from .activation import act_fn


def finfo(dtype):
    if dtype == mstype.float32:
        return Tensor(np.finfo(np.float32).min, mstype.float32)
    elif dtype == mstype.float16:
        return Tensor(np.finfo(np.float16).min, mstype.float16)
    else:
        raise TypeError(f"For 'finfo', the input dtype should be float32 or float16, bug got {dtype}")


class LayoutXLMEmbeddings(nn.Cell):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids = Parameter(
            Tensor(np.arange(0, config.max_position_embeddings)).broadcast_to((1, -1)),
            name="position_ids",
            requires_grad=False,
        )

    def _cal_spatial_position_embeddings(self, bbox):
        bbox_0 = bbox[:, :, 0]
        bbox_1 = bbox[:, :, 1]
        bbox_2 = bbox[:, :, 2]
        bbox_3 = bbox[:, :, 3]
        left_position_embeddings = self.x_position_embeddings(bbox_0)
        upper_position_embeddings = self.y_position_embeddings(bbox_1)
        right_position_embeddings = self.x_position_embeddings(bbox_2)
        lower_position_embeddings = self.y_position_embeddings(bbox_3)

        h_position_embeddings = self.h_position_embeddings(bbox_3 - bbox_1)
        w_position_embeddings = self.w_position_embeddings(bbox_2 - bbox_0)

        spatial_position_embeddings = ops.concat(
            (
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ),
            axis=-1,
        )
        return spatial_position_embeddings

    def construct(self, input_ids, bbox=None, token_type_ids=None, position_ids=None):
        raise NotImplementedError(
            f"'construct' is not implemented for {self.__class__}. "
            f"For implement it, you should overwrite this method."
        )


class LayoutXLMSelfAttention(nn.Cell):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size {} is not a multiple of the number of attention "
                "heads {}".format(config.hidden_size, config.num_attention_heads)
            )
        self.fast_qkv = config.fast_qkv
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attention_head_size_sqrt = math.sqrt(self.attention_head_size)

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        self.use_float16 = config.use_float16
        self.dense_dtype = mstype.float32
        if self.use_float16 is True:
            self.dense_dtype = mstype.float16

        if config.fast_qkv:
            self.qkv_linear = nn.Dense(config.hidden_size, 3 * self.all_head_size, has_bias=False).to_float(
                self.dense_dtype
            )
            self.q_bias = Parameter(initializer(Constant(0.0), [1, 1, self.all_head_size], self.dense_dtype))
            self.v_bias = Parameter(initializer(Constant(0.0), [1, 1, self.all_head_size], self.dense_dtype))
        else:
            self.query = nn.Dense(config.hidden_size, self.all_head_size).to_float(self.dense_dtype)
            self.key = nn.Dense(config.hidden_size, self.all_head_size).to_float(self.dense_dtype)
            self.value = nn.Dense(config.hidden_size, self.all_head_size).to_float(self.dense_dtype)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.min = finfo(self.dense_dtype)

    def transpose_for_scores(self, x):
        new_x_shape = list(x.shape[:-1]) + [
            self.num_attention_heads,
            self.attention_head_size,
        ]

        x = x.reshape(tuple(new_x_shape))
        return x.transpose((0, 2, 1, 3))

    def compute_qkv(self, hidden_states):
        if self.fast_qkv:
            qkv = self.qkv_linear(hidden_states)
            q, k, v = ops.chunk(qkv, 3, axis=-1)
            if q.ndimension() == self.q_bias.ndimension():
                q = q + self.q_bias
                v = v + self.v_bias
            else:
                _sz = (1,) * (q.ndimension() - 1) + (-1,)
                q = q + self.q_bias.reshape(_sz)
                v = v + self.v_bias.reshape(_sz)
        else:
            q = self.query(hidden_states)
            k = self.key(hidden_states)
            v = self.value(hidden_states)
        return q, k, v

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        q, k, v = self.compute_qkv(hidden_states)

        # (B, L, H*D) -> (B, H, L, D)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        query_layer = query_layer / self.attention_head_size_sqrt
        # [BSZ, NAT, L, L]
        attention_scores = ops.matmul(
            query_layer,
            key_layer.transpose((0, 1, 3, 2)),
        )
        if self.has_relative_attention_bias:
            attention_scores += rel_pos
        if self.has_spatial_attention_bias:
            attention_scores += rel_2d_pos
        attention_scores = ops.masked_fill(
            attention_scores,
            ops.stop_gradient(attention_mask.astype(mstype.bool_)),
            self.min,
        )
        attention_probs = ops.softmax(attention_scores, axis=-1)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = ops.matmul(attention_probs, value_layer)

        context_layer = context_layer.transpose((0, 2, 1, 3))
        new_context_layer_shape = list(context_layer.shape[:-2]) + [self.all_head_size]
        context_layer = context_layer.reshape(new_context_layer_shape)

        if output_attentions:
            outputs = [context_layer, attention_probs]
        else:
            outputs = [context_layer]
        return outputs


class LayoutXLMSelfOutput(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.use_float16 = config.use_float16
        self.dense_dtype = mstype.float32
        if self.use_float16 is True:
            self.dense_dtype = mstype.float16
        self.dense = nn.Dense(config.hidden_size, config.hidden_size).to_float(self.dense_dtype)
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayoutXLMAttention(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.self_attention = LayoutXLMSelfAttention(config)
        self.output = LayoutXLMSelfOutput(config)

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        self_outputs = self.self_attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        # add attentions if we output them
        if output_attentions:
            outputs = [
                attention_output,
            ] + self_outputs[1:]
        else:
            outputs = [attention_output]
        return outputs


class LayoutXLMIntermediate(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.use_float16 = config.use_float16
        self.dense_dtype = mstype.float32
        if self.use_float16 is True:
            self.dense_dtype = mstype.float16
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size).to_float(self.dense_dtype)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = act_fn[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class LayoutXLMOutput(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.use_float16 = config.use_float16
        self.dense_dtype = mstype.float32
        if self.use_float16 is True:
            self.dense_dtype = mstype.float16
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size).to_float(self.dense_dtype)
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayoutXLMLayer(nn.Cell):
    def __init__(self, config):
        super().__init__(config)
        # since chunk_size_feed_forward is 0 as default, no chunk is needed here.
        self.seq_len_dim = 1
        self.attention = LayoutXLMAttention(config)
        self.add_cross_attention = False  # default as false
        self.intermediate = LayoutXLMIntermediate(config)
        self.output = LayoutXLMOutput(config)

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self_attention_outputs[0]
        layer_output = self.feed_forward_chunk(attention_output)

        if output_attentions:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
            outputs = [
                layer_output,
            ] + list(outputs)
        else:
            outputs = [layer_output]
        return outputs


class LayoutXLMEncoder(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.CellList([LayoutXLMLayer(config) for _ in range(config.num_hidden_layers)])

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        self.use_float16 = config.use_float16
        self.dense_dtype = mstype.float32
        if self.use_float16 is True:
            self.dense_dtype = mstype.float16

        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_onehot_size = config.rel_pos_bins
            self.rel_pos_bias = nn.Dense(self.rel_pos_onehot_size, config.num_attention_heads, has_bias=False).to_float(
                mstype.float16
            )

        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_2d_pos_onehot_size = config.rel_2d_pos_bins
            self.rel_pos_x_bias = nn.Dense(
                self.rel_2d_pos_onehot_size, config.num_attention_heads, has_bias=False
            ).to_float(self.dense_dtype)
            self.rel_pos_y_bias = nn.Dense(
                self.rel_2d_pos_onehot_size, config.num_attention_heads, has_bias=False
            ).to_float(self.dense_dtype)

    def relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        def test(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
            ret = 0
            if bidirectional:
                num_buckets //= 2
                ret += (relative_position > 0).astype(mstype.int64) * num_buckets
                n = ops.abs(relative_position)
            else:
                n = ops.maximum(-relative_position, ops.zeros_like(relative_position))  # to be confirmed
            # Now n is in the range [0, inf)
            # half of the buckets are for exact increments in positions
            max_exact = num_buckets // 2
            is_small = n < max_exact

            # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
            scaling_val = ops.log(n.astype(mstype.float32) / max_exact) / math.log(max_distance / max_exact)
            scaling_val = scaling_val * (num_buckets - max_exact)
            val_if_large = max_exact + scaling_val.astype(mstype.int64)

            val_if_large = ops.minimum(val_if_large, ops.full_like(val_if_large, num_buckets - 1))

            ret += ops.where(is_small, n, val_if_large)
            return ret

        # test(relative_position.copy(), num_buckets=num_buckets, max_distance=max_distance)
        ret = 0
        if bidirectional:
            num_buckets //= 2
            ret += (relative_position > 0).long() * num_buckets
            n = ops.abs(relative_position)
        else:
            n = ops.maximum(-relative_position, ops.zeros_like(relative_position))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            ops.log(n.astype(mstype.float32) / max_exact)
            / ops.log(Tensor(max_distance / max_exact))
            * (num_buckets - max_exact)
        ).astype(mstype.int64)
        val_if_large = ops.minimum(val_if_large, ops.full_like(val_if_large, num_buckets - 1))

        ret += ops.where(is_small, n, val_if_large)
        return ret

    def _cal_1d_pos_emb(self, hidden_states, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = self.relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        on_value, off_value = Tensor(1.0, mstype.float32), Tensor(0.0, mstype.float32)
        rel_pos = ops.one_hot(rel_pos, self.rel_pos_onehot_size, on_value, off_value).astype(hidden_states.dtype)
        rel_pos = self.rel_pos_bias(rel_pos).transpose((0, 3, 1, 2))
        return rel_pos

    def _cal_2d_pos_emb(self, hidden_states, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        rel_pos_x = self.relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = self.relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        on_value, off_value = Tensor(1.0, mstype.float32), Tensor(0.0, mstype.float32)
        rel_pos_x = ops.one_hot(rel_pos_x, self.rel_2d_pos_onehot_size, on_value, off_value).astype(hidden_states.dtype)
        rel_pos_y = ops.one_hot(rel_pos_y, self.rel_2d_pos_onehot_size, on_value, off_value).astype(hidden_states.dtype)
        rel_pos_x = self.rel_pos_x_bias(rel_pos_x).transpose((0, 3, 1, 2))
        rel_pos_y = self.rel_pos_y_bias(rel_pos_y).transpose((0, 3, 1, 2))
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
        bbox=None,
        position_ids=None,
    ):
        all_hidden_states = () if output_hidden_states else None

        rel_pos = self._cal_1d_pos_emb(hidden_states, position_ids) if self.has_relative_attention_bias else None
        rel_2d_pos = self._cal_2d_pos_emb(hidden_states, bbox) if self.has_spatial_attention_bias else None

        hidden_save = dict()
        hidden_save["input_hidden_states"] = hidden_states

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = None
            past_key_value = None
            # gradient_checkpointing is set as False here so we remove some codes here
            hidden_save["input_attention_mask"] = attention_mask
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
                rel_pos=rel_pos,
                rel_2d_pos=rel_2d_pos,
            )

            hidden_states = layer_outputs[0]

            hidden_save["{}_data".format(i)] = hidden_states

        return hidden_states, hidden_save


class LayoutXLMPooler(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.use_float16 = config.use_float16
        self.dense_dtype = mstype.float32
        if self.use_float16 is True:
            self.dense_dtype = mstype.float16
        self.dense = nn.Dense(config.hidden_size, config.hidden_size).to_float(self.dense_dtype)
        self.activation = nn.Tanh()

    def construct(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
