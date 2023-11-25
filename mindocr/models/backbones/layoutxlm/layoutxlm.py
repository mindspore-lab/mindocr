import math

import numpy as np

import mindspore as ms
from mindspore import Parameter, nn, ops
from mindspore.common.initializer import Constant, initializer

from .._registry import register_backbone, register_backbone_class
from ..mindcv_models.utils import load_pretrained
from .configuration import LayoutXLMPretrainedConfig
from .visual_backbone import build_resnet_fpn_backbone, read_config


def _cfg(url="", use_visual_backbone=True, **kwargs):
    return {
        "url": url,
        "use_visual_backbone": use_visual_backbone,
        **kwargs,
    }


default_cfgs = {
    "layoutxlm": _cfg(
        url="https://download.mindspore.cn/toolkits/mindocr/vi-layoutxlm/ser_vi_layoutxlm-1e740bcb.ckpt",
        use_visual_backbone=True,
    ),
    "vi-layoutxlm": _cfg(
        url="https://download.mindspore.cn/toolkits/mindocr/vi-layoutxlm/ser_vi_layoutxlm-1e740bcb.ckpt",
        use_visual_backbone=False,
    ),
}


class VisualBackbone(nn.Cell):
    def __init__(self, config):
        super(VisualBackbone, self).__init__()
        self.cfg = read_config()
        self.backbone = build_resnet_fpn_backbone(self.cfg)

        if len(self.cfg.MODEL.PIXEL_MEAN) != len(self.cfg.MODEL.PIXEL_STD):
            raise ValueError(
                "cfg.model.pixel_mean is not equal with cfg.model.pixel_std."
            )
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)

        self.pixel_mean = Parameter(
            ms.Tensor(self.cfg.MODEL.PIXEL_MEAN).reshape((num_channels, 1, 1)),
            name="pixel_mean",
            requires_grad=False,
        )
        self.pixel_std = Parameter(
            ms.Tensor(self.cfg.MODEL.PIXEL_STD).reshape((num_channels, 1, 1)),
            name="pixel_std",
            requires_grad=False,
        )

        self.out_feature_key = "p2"
        self.pool_shape = tuple(config.image_feature_pool_shape[:2])  # (7,7)
        if len(config.image_feature_pool_shape) == 2:
            config.image_feature_pool_shape.append(
                self.backbone.output_shape()[self.out_feature_key].channels
            )

    def construct(self, images):
        images_input = (images - self.pixel_mean) / self.pixel_std
        features = self.backbone(images_input)
        for item in features:
            if item[0] == self.out_feature_key:
                features = item[1]
        channel = features.shape[1]
        weight = ms.Tensor(np.ones([channel, channel, 1, 1]), dtype=ms.float32)
        features = (
            ops.conv2d(features, weight, stride=self.pool_shape[0] + 1)
            .flatten(start_dim=2)
            .transpose((0, 2, 1))
        )
        return features


def relative_position_bucket(
    relative_position, bidirectional=True, num_buckets=32, max_distance=128
):
    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (relative_position > 0).astype(ms.int64) * num_buckets
        n = ops.abs(relative_position)
    else:
        n = ops.maximum(
            -relative_position, ops.zeros_like(relative_position)
        )  # to be confirmed
    # Now n is in the range [0, inf)
    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        ops.log(n.astype(ms.float32) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(ms.int64)

    val_if_large = ops.minimum(
        val_if_large, ops.full_like(val_if_large, num_buckets - 1)
    )

    ret += ops.where(is_small, n, val_if_large)
    return ret


class LayoutXLMEmbeddings(nn.Cell):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self, config):
        super(LayoutXLMEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        self.x_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.coordinate_size
        )
        self.y_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.coordinate_size
        )
        self.h_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.shape_size
        )
        self.w_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.shape_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.LayerNorm = nn.LayerNorm(
            (config.hidden_size,), epsilon=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.position_ids = Parameter(
            ms.Tensor(np.arange(0, config.max_position_embeddings)).broadcast_to(
                (1, -1)
            ),
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
        if position_ids is None:
            ones = ops.ones_like(input_ids, dtype=ms.int64)
            seq_length = ops.cumsum(ones, axis=-1)

            position_ids = seq_length - ones
            position_ids = ops.stop_gradient(
                position_ids
            )  # position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids, dtype=ms.int64)

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        h_position_embeddings = self.h_position_embeddings(
            bbox[:, :, 3] - bbox[:, :, 1]
        )
        w_position_embeddings = self.w_position_embeddings(
            bbox[:, :, 2] - bbox[:, :, 0]
        )

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (
            input_embedings
            + position_embeddings
            + left_position_embeddings
            + upper_position_embeddings
            + right_position_embeddings
            + lower_position_embeddings
            + h_position_embeddings
            + w_position_embeddings
            + token_type_embeddings
        )

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LayoutXLMSelfAttention(nn.Cell):
    def __init__(self, config):
        super(LayoutXLMSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
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
        self.dense_dtype = ms.float32
        if self.use_float16 is True:
            self.dense_dtype = ms.float16

        if config.fast_qkv:
            self.qkv_linear = nn.Dense(
                config.hidden_size, 3 * self.all_head_size, has_bias=False
            ).to_float(self.dense_dtype)
            self.q_bias = Parameter(
                initializer(Constant(0.0), [1, 1, self.all_head_size], ms.float32)
            )
            self.v_bias = Parameter(
                initializer(Constant(0.0), [1, 1, self.all_head_size], ms.float32)
            )
        else:
            self.query = nn.Dense(config.hidden_size, self.all_head_size).to_float(
                self.dense_dtype
            )
            self.key = nn.Dense(config.hidden_size, self.all_head_size).to_float(
                self.dense_dtype
            )
            self.value = nn.Dense(config.hidden_size, self.all_head_size).to_float(
                self.dense_dtype
            )

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

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
            query_layer.astype(ms.float16),
            key_layer.transpose((0, 1, 3, 2)).astype(ms.float16),
        ).astype(ms.float32)
        if self.has_relative_attention_bias:
            attention_scores += rel_pos
        if self.has_spatial_attention_bias:
            attention_scores += rel_2d_pos
        bool_attention_mask = attention_mask.bool()  # ms.int32 or ms.bool
        bool_attention_mask = ops.stop_gradient(bool_attention_mask)
        attention_scores_shape = ops.shape(attention_scores)
        attention_scores = ops.where(
            bool_attention_mask.broadcast_to(attention_scores_shape),
            ops.ones(attention_scores_shape) * float("-1e10"),
            attention_scores,
        )
        attention_probs = ops.softmax(attention_scores, axis=-1)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)
        context_layer = ops.matmul(
            attention_probs.astype(ms.float16), value_layer.astype(ms.float16)
        ).astype(ms.float32)

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
        super(LayoutXLMSelfOutput, self).__init__()
        self.use_float16 = config.use_float16
        self.dense_dtype = ms.float32
        if self.use_float16 is True:
            self.dense_dtype = ms.float16
        self.dense = nn.Dense(config.hidden_size, config.hidden_size).to_float(
            self.dense_dtype
        )
        self.LayerNorm = nn.LayerNorm(
            (config.hidden_size,), epsilon=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayoutXLMAttention(nn.Cell):
    def __init__(self, config):
        super(LayoutXLMAttention, self).__init__()
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
        super(LayoutXLMIntermediate, self).__init__()
        self.use_float16 = config.use_float16
        self.dense_dtype = ms.float32
        if self.use_float16 is True:
            self.dense_dtype = ms.float16
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size).to_float(
            self.dense_dtype
        )
        if config.hidden_act == "gelu":
            self.intermediate_act_fn = nn.GELU()
        else:
            raise ValueError(
                "hidden_act is set as: {}, please check it..".format(config.hidden_act)
            )

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class LayoutXLMOutput(nn.Cell):
    def __init__(self, config):
        super(LayoutXLMOutput, self).__init__()
        self.use_float16 = config.use_float16
        self.dense_dtype = ms.float32
        if self.use_float16 is True:
            self.dense_dtype = ms.float16
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size).to_float(
            self.dense_dtype
        )
        self.LayerNorm = nn.LayerNorm(
            (config.hidden_size,), epsilon=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayoutXLMLayer(nn.Cell):
    def __init__(self, config):
        super(LayoutXLMLayer, self).__init__()
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
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
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
            outputs = self_attention_outputs[
                1:
            ]  # add self attentions if we output attention weights
            outputs = [
                layer_output,
            ] + list(outputs)
        else:
            outputs = [layer_output]
        return outputs


class LayoutXLMEncoder(nn.Cell):
    def __init__(self, config):
        super(LayoutXLMEncoder, self).__init__()
        self.config = config
        self.layer = nn.CellList(
            [LayoutXLMLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        self.use_float16 = config.use_float16
        self.dense_dtype = ms.float32
        if self.use_float16 is True:
            self.dense_dtype = ms.float16

        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_onehot_size = config.rel_pos_bins
            self.rel_pos_bias = nn.Dense(
                self.rel_pos_onehot_size, config.num_attention_heads, has_bias=False
            ).to_float(ms.float16)

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

    def _cal_1d_pos_emb(self, hidden_states, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        on_value, off_value = ms.Tensor(1.0, ms.float32), ms.Tensor(0.0, ms.float32)
        rel_pos = ops.one_hot(
            rel_pos, self.rel_pos_onehot_size, on_value, off_value
        ).astype(hidden_states.dtype)
        rel_pos = self.rel_pos_bias(rel_pos).transpose((0, 3, 1, 2))
        return rel_pos

    def _cal_2d_pos_emb(self, hidden_states, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(
            -1
        )
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(
            -1
        )
        rel_pos_x = relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        on_value, off_value = ms.Tensor(1.0, ms.float32), ms.Tensor(0.0, ms.float32)
        rel_pos_x = ops.one_hot(
            rel_pos_x, self.rel_2d_pos_onehot_size, on_value, off_value
        ).astype(hidden_states.dtype)
        rel_pos_y = ops.one_hot(
            rel_pos_y, self.rel_2d_pos_onehot_size, on_value, off_value
        ).astype(hidden_states.dtype)
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

        rel_pos = (
            self._cal_1d_pos_emb(hidden_states, position_ids)
            if self.has_relative_attention_bias
            else None
        )
        rel_2d_pos = (
            self._cal_2d_pos_emb(hidden_states, bbox)
            if self.has_spatial_attention_bias
            else None
        )

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
        super(LayoutXLMPooler, self).__init__()
        self.use_float16 = config.use_float16
        self.dense_dtype = ms.float32
        if self.use_float16 is True:
            self.dense_dtype = ms.float16
        self.dense = nn.Dense(config.hidden_size, config.hidden_size).to_float(
            self.dense_dtype
        )
        self.activation = nn.Tanh()

    def construct(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@register_backbone_class
class LayoutXLMModel(nn.Cell):
    def __init__(self, config):
        super(LayoutXLMModel, self).__init__()
        self.config = config
        self.has_visual_segment_embedding = config.has_visual_segment_embedding
        self.embeddings = LayoutXLMEmbeddings(config)
        self.use_visual_backbone = config.use_visual_backbone
        self.use_float16 = config.use_float16
        self.dense_dtype = ms.float32
        if self.use_float16 is True:
            self.dense_dtype = ms.float16

        if self.use_visual_backbone is True:
            self.visual = VisualBackbone(config)
            self.visual_proj = nn.Dense(
                config.image_feature_pool_shape[-1], config.hidden_size
            ).to_float(self.dense_dtype)
        if self.has_visual_segment_embedding:
            self.visual_segment_embedding = Parameter(
                nn.Embedding(1, config.hidden_size).embedding_table[0]
            )
        self.visual_LayerNorm = nn.LayerNorm(
            (config.hidden_size,), epsilon=config.layer_norm_eps
        )
        self.visual_dropout = nn.Dropout(p=config.hidden_dropout_prob)

        self.encoder = LayoutXLMEncoder(config)
        self.pooler = LayoutXLMPooler(config)
        self.image_feature_pool_shape_size = (
            config.image_feature_pool_shape[0] * config.image_feature_pool_shape[1]
        )
        self.image_feature_pool_shape = config.image_feature_pool_shape
        self.num_hidden_layers = config.num_hidden_layers
        self.max_position_embeddings = config.max_position_embeddings
        self.hidden_size = config.hidden_size
        self.out_channels = 1

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _calc_text_embeddings(self, input_ids, bbox, position_ids, token_type_ids):
        words_embeddings = self.embeddings.word_embeddings(input_ids)
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(
            bbox
        )
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)
        embeddings = (
            words_embeddings
            + position_embeddings
            + spatial_position_embeddings
            + token_type_embeddings
        )
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings

    def _calc_img_embeddings(self, image, bbox, position_ids):
        use_image_info = self.use_visual_backbone and image is not None
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(
            bbox
        )
        if use_image_info:
            visual_embeddings = self.visual_proj(self.visual(image))
            embeddings = (
                visual_embeddings + position_embeddings + spatial_position_embeddings
            )
        else:
            embeddings = position_embeddings + spatial_position_embeddings
        if self.has_visual_segment_embedding:
            embeddings += self.visual_segment_embedding
        embeddings = self.visual_LayerNorm(embeddings)
        embeddings = self.visual_dropout(embeddings)
        return embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config["max_position_embeddings"]`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end.
        """
        num_position_embeds_diff = (
            new_num_position_embeddings - self.max_position_embeddings
        )

        # no resizing needs to be done if the length stays the same
        if num_position_embeds_diff == 0:
            return

        self.max_position_embeddings = new_num_position_embeddings

        old_position_embeddings_weight = (
            self.embeddings.position_embeddings.embedding_table
        )

        self.embeddings.position_embeddings = nn.Embedding(
            self.max_position_embeddings, self.hidden_size
        )

        if num_position_embeds_diff > 0:
            self.embeddings.position_embeddings.embedding_table[
                :-num_position_embeds_diff
            ] = old_position_embeddings_weight
        else:
            self.embeddings.position_embeddings.embedding_table = (
                old_position_embeddings_weight[:num_position_embeds_diff]
            )

    def _calc_visual_bbox(self, image_feature_pool_shape, bbox, visual_shape):
        x_size = image_feature_pool_shape[1]
        y_size = image_feature_pool_shape[0]
        visual_bbox_x = ms.Tensor(
            np.arange(0, 1000 * (x_size + 1), 1000) // x_size, dtype=ms.int64
        )
        visual_bbox_y = ms.Tensor(
            np.arange(0, 1000 * (y_size + 1), 1000) // y_size, dtype=ms.int64
        )
        expand_shape = image_feature_pool_shape[0:2]
        expand_shape = tuple(expand_shape)
        visual_bbox = ops.stack(
            [
                visual_bbox_x[:-1].broadcast_to(expand_shape),
                visual_bbox_y[:-1].broadcast_to(expand_shape[::-1]).transpose((1, 0)),
                visual_bbox_x[1:].broadcast_to(expand_shape),
                visual_bbox_y[1:].broadcast_to(expand_shape[::-1]).transpose((1, 0)),
            ],
            axis=-1,
        ).reshape((expand_shape[0] * expand_shape[1], ops.shape(bbox)[-1]))
        visual_bbox = visual_bbox.broadcast_to(
            (visual_shape[0], visual_bbox.shape[0], visual_bbox.shape[1])
        )
        return visual_bbox

    def _get_input_shape(self, input_ids=None, inputs_embeds=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            return input_ids.shape
        elif inputs_embeds is not None:
            return inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

    def construct(
        self,
        input_ids=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        input_shape = self._get_input_shape(input_ids, inputs_embeds)
        visual_shape = list(input_shape)
        visual_shape[1] = self.image_feature_pool_shape_size
        visual_bbox = self._calc_visual_bbox(
            self.image_feature_pool_shape, bbox, visual_shape
        )

        final_bbox = ops.concat([bbox, visual_bbox], axis=1)
        if attention_mask is None:
            attention_mask = ops.ones(input_shape)

        if self.use_visual_backbone is True:
            visual_attention_mask = ops.ones(visual_shape)
        else:
            visual_attention_mask = ops.zeros(visual_shape)

        attention_mask = attention_mask.astype(visual_attention_mask.dtype)

        final_attention_mask = ops.concat(
            [attention_mask, visual_attention_mask], axis=1
        )

        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, dtype=ms.int64)

        if position_ids is None:
            seq_length = input_shape[1]
            position_ids = self.embeddings.position_ids[:, :seq_length]
            position_ids = position_ids.broadcast_to(input_shape)

        visual_position_ids = ms.Tensor(np.arange(0, visual_shape[1])).broadcast_to(
            (input_shape[0], visual_shape[1])
        )
        final_position_ids = ops.concat([position_ids, visual_position_ids], axis=1)

        if bbox is None:
            bbox = ops.zeros(input_shape + [4])
        text_layout_emb = self._calc_text_embeddings(
            input_ids=input_ids,
            bbox=bbox,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        visual_emb = self._calc_img_embeddings(
            image=image,
            bbox=visual_bbox,
            position_ids=visual_position_ids,
        )
        final_emb = ops.concat([text_layout_emb, visual_emb], axis=1)

        extended_attention_mask = final_attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.broadcast_to(
                    (self.num_hidden_layers, -1, -1, -1, -1)
                )
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        else:
            head_mask = [None] * self.num_hidden_layers

        encoder_outputs = self.encoder(
            final_emb,
            extended_attention_mask,
            bbox=final_bbox,
            position_ids=final_position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output, encoder_outputs[1]


@register_backbone
def layoutxlm(
    pretrained: bool = True,
    use_visual_backbone: bool = True,
    use_float16: bool = False,
    **kwargs
):
    pretrained_config = LayoutXLMPretrainedConfig(use_visual_backbone, use_float16)
    model = LayoutXLMModel(pretrained_config)
    if pretrained:
        if use_visual_backbone is True:
            default_cfg = default_cfgs["layoutxlm"]
        else:
            default_cfg = default_cfgs["vi-layoutxlm"]
        load_pretrained(model, default_cfg)
    return model
