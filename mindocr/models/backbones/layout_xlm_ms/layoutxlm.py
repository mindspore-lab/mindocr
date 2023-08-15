import copy
import math
import os

import numpy as np

import mindspore as ms
from mindspore import Parameter, load_checkpoint, load_param_into_net, nn, ops
from mindspore.common.initializer import Constant, initializer

from .._registry import register_backbone, register_backbone_class
from .configuration import LayoutXLMPretrainedConfig
from .visual_backbone import build_resnet_fpn_backbone, read_config

os.environ["MS_DEV_JIT_SYNTAX_LEVEL"] = "0"


class VisualBackbone(nn.Cell):
    def __init__(self, config):
        super(VisualBackbone, self).__init__()
        self.cfg = read_config()
        self.backbone = build_resnet_fpn_backbone(self.cfg)

        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)

        self.pixel_mean = Parameter(
            ms.Tensor(self.cfg.MODEL.PIXEL_MEAN).reshape((num_channels, 1, 1)),
            name="pixel_mean", requires_grad=False)
        self.pixel_std = Parameter(
            ms.Tensor(self.cfg.MODEL.PIXEL_STD).reshape((num_channels, 1, 1)),
            name="pixel_std", requires_grad=False)

        self.out_feature_key = "p2"
        self.pool_shape = tuple(config.image_feature_pool_shape[:2])  # (7,7)
        if len(config.image_feature_pool_shape) == 2:
            config.image_feature_pool_shape.append(self.backbone.output_shape()[self.out_feature_key].channels)
        assert self.backbone.output_shape()[self.out_feature_key].channels == config.image_feature_pool_shape[2]

    def construct(self, images):
        images_input = (images - self.pixel_mean) / self.pixel_std
        features = self.backbone(images_input)
        for item in features:
            if item[0] == self.out_feature_key:
                features = item[1]
        channel = features.shape[1]
        weight = ms.Tensor(np.ones([channel, channel, 1, 1]), dtype=ms.float32)
        features = ops.conv2d(features, weight, stride=self.pool_shape[0] + 1).flatten(
            start_dim=2).transpose((0, 2, 1))
        return features


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (relative_position > 0).astype(ms.int64) * num_buckets
        n = ops.abs(relative_position)
    else:
        n = ops.maximum(-relative_position, ops.zeros_like(relative_position))  # to be confirmed
    # Now n is in the range [0, inf)
    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
            ops.log(n.astype(ms.float32) / max_exact) \
            / math.log(max_distance / max_exact) \
            * (num_buckets - max_exact)
    ).astype(ms.int64)

    val_if_large = ops.minimum(val_if_large, ops.full_like(val_if_large, num_buckets - 1))

    ret += ops.where(is_small, n, val_if_large)
    return ret


class LayoutXLMEmbeddings(nn.Cell):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self, config):
        super(LayoutXLMEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.position_ids = Parameter(
            ms.Tensor(np.arange(0, config.max_position_embeddings)).broadcast_to((1, -1)),
            name="position_ids", requires_grad=False)

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
            position_ids = ops.stop_gradient(position_ids)  # position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids, dtype=ms.int64)

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The :obj:`bbox`coordinate values should be within 0-1000 range.") from e
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

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

        if config.fast_qkv:
            self.qkv_linear = nn.Dense(config.hidden_size, 3 * self.all_head_size, has_bias=False)
            self.q_bias = Parameter(
                initializer(Constant(0.0), [1, 1, self.all_head_size], ms.float32)
            )
            self.v_bias = Parameter(
                initializer(Constant(0.0), [1, 1, self.all_head_size], ms.float32)
            )
        else:
            self.query = nn.Dense(config.hidden_size, self.all_head_size)
            self.key = nn.Dense(config.hidden_size, self.all_head_size)
            self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = list(x.shape[:-1]) + [self.num_attention_heads, self.attention_head_size]

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
        attention_scores = ops.matmul(query_layer.astype(ms.float16),
                                      key_layer.transpose((0, 1, 3, 2)).astype(ms.float16)).astype(ms.float32)
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
        context_layer = ops.matmul(attention_probs.astype(ms.float16), value_layer.astype(ms.float16)).astype(
            ms.float32)

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
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
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
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if config.hidden_act == "gelu":
            self.intermediate_act_fn = nn.GELU()
        else:
            assert False, "hidden_act is set as: {}, please check it..".format(config.hidden_act)

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class LayoutXLMOutput(nn.Cell):
    def __init__(self, config):
        super(LayoutXLMOutput, self).__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
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
        super(LayoutXLMEncoder, self).__init__()
        self.config = config
        self.layer = nn.CellList([LayoutXLMLayer(config) for _ in range(config.num_hidden_layers)])

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_onehot_size = config.rel_pos_bins
            self.rel_pos_bias = nn.Dense(self.rel_pos_onehot_size, config.num_attention_heads, has_bias=False).to_float(
                ms.float16)

        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_2d_pos_onehot_size = config.rel_2d_pos_bins
            self.rel_pos_x_bias = nn.Dense(self.rel_2d_pos_onehot_size, config.num_attention_heads,
                                           has_bias=False)
            self.rel_pos_y_bias = nn.Dense(self.rel_2d_pos_onehot_size, config.num_attention_heads,
                                           has_bias=False)

    def _cal_1d_pos_emb(self, hidden_states, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        on_value, off_value = ms.Tensor(1.0, ms.float32), ms.Tensor(0.0, ms.float32)
        rel_pos = ops.one_hot(rel_pos,
                              self.rel_pos_onehot_size,
                              on_value,
                              off_value).astype(
            hidden_states.dtype
        )
        rel_pos = self.rel_pos_bias(rel_pos).transpose((0, 3, 1, 2))
        return rel_pos

    def _cal_2d_pos_emb(self, hidden_states, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
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
        super(LayoutXLMPooler, self).__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LayoutXLMModel(nn.Cell):
    def __init__(self, config):
        super(LayoutXLMModel, self).__init__()
        self.config = config
        self.has_visual_segment_embedding = config.has_visual_segment_embedding
        self.embeddings = LayoutXLMEmbeddings(config)

        self.visual = VisualBackbone(config)
        self.visual_proj = nn.Dense(config.image_feature_pool_shape[-1], config.hidden_size)
        if self.has_visual_segment_embedding:
            self.visual_segment_embedding = Parameter(nn.Embedding(1, config.hidden_size).embedding_table[0])
        self.visual_LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.visual_dropout = nn.Dropout(p=config.hidden_dropout_prob)

        self.encoder = LayoutXLMEncoder(config)
        self.pooler = LayoutXLMPooler(config)
        self.image_feature_pool_shape_size = config.image_feature_pool_shape[0] * config.image_feature_pool_shape[1]
        self.image_feature_pool_shape = config.image_feature_pool_shape
        self.num_hidden_layers = config.num_hidden_layers
        self.max_position_embeddings = config.max_position_embeddings
        self.hidden_size = config.hidden_size

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _calc_text_embeddings(self, input_ids, bbox, position_ids, token_type_ids):
        words_embeddings = self.embeddings.word_embeddings(input_ids)
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(bbox)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + spatial_position_embeddings + token_type_embeddings
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings

    def _calc_img_embeddings(self, image, bbox, position_ids):
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(bbox)
        visual_embeddings = self.visual_proj(self.visual(image.astype(ms.float32)))
        embeddings = visual_embeddings + position_embeddings + spatial_position_embeddings
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
        num_position_embeds_diff = new_num_position_embeddings - self.max_position_embeddings

        # no resizing needs to be done if the length stays the same
        if num_position_embeds_diff == 0:
            return

        self.max_position_embeddings = new_num_position_embeddings

        old_position_embeddings_weight = self.embeddings.position_embeddings.embedding_table

        self.embeddings.position_embeddings = nn.Embedding(
            self.max_position_embeddings, self.hidden_size
        )

        if num_position_embeds_diff > 0:
            self.embeddings.position_embeddings.embedding_table[:-num_position_embeds_diff] = \
                old_position_embeddings_weight
        else:
            self.embeddings.position_embeddings.embedding_table = \
                old_position_embeddings_weight[:num_position_embeds_diff]

    def _calc_visual_bbox(self, image_feature_pool_shape, bbox, visual_shape):
        visual_bbox_x = (ms.Tensor(
            np.arange(
                0,
                1000 * (image_feature_pool_shape[1] + 1),
                1000,
            ) // image_feature_pool_shape[1],
            dtype=ms.int64
        )
        )
        visual_bbox_y = (ms.Tensor(
            np.arange(
                0,
                1000 * (image_feature_pool_shape[0] + 1),
                1000,
            ) // image_feature_pool_shape[0],
            dtype=ms.int64,
        )
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
        visual_bbox = visual_bbox.broadcast_to((visual_shape[0], visual_bbox.shape[0], visual_bbox.shape[1]))
        return visual_bbox

    def _get_input_shape(self, input_ids=None, inputs_embeds=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
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
            output_hidden_states=False
    ):
        input_shape = self._get_input_shape(input_ids, inputs_embeds)
        visual_shape = list(input_shape)
        visual_shape[1] = self.image_feature_pool_shape_size
        visual_bbox = self._calc_visual_bbox(self.image_feature_pool_shape, bbox, visual_shape)

        final_bbox = ops.concat([bbox, visual_bbox], axis=1)
        if attention_mask is None:
            attention_mask = ops.ones(input_shape)

        visual_attention_mask = ops.ones(visual_shape)
        attention_mask = attention_mask.astype(visual_attention_mask.dtype)

        final_attention_mask = ops.concat([attention_mask, visual_attention_mask], axis=1)

        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, dtype=ms.int64)

        if position_ids is None:
            seq_length = input_shape[1]
            position_ids = self.embeddings.position_ids[:, :seq_length]
            position_ids = position_ids.broadcast_to(input_shape)

        visual_position_ids = ms.Tensor(np.arange(0, visual_shape[1])).broadcast_to((input_shape[0], visual_shape[1]))
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
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.broadcast_to((self.num_hidden_layers, -1, -1, -1, -1))
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


class LayoutXLMForTokenClassification(nn.Cell):
    def __init__(self, layoutxlm, num_classes=2, dropout=None):
        super(LayoutXLMForTokenClassification, self).__init__()
        self.num_classes = num_classes
        if isinstance(layoutxlm, dict):
            self.layoutxlm = LayoutXLMModel(**layoutxlm)
        else:
            self.layoutxlm = layoutxlm
        dropout_prob = dropout if dropout is not None else self.layoutxlm.config.hidden_dropout_prob
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Dense(self.layoutxlm.config.hidden_size, num_classes)
        self._init_weights(self.classifier)

    @staticmethod
    def _init_weights(layer):
        """Initialize the weights"""
        if isinstance(layer, nn.Dense):
            layer.weight.set_data(ops.normal(
                shape=layer.weight.shape,
                mean=0.0,
                stddev=0.02
            ))
            if layer.bias is not None:
                layer.bias.set_data(ops.zeros(size=layer.bias.shape))

    def get_input_embeddings(self):
        return self.layoutxlm.embeddings.word_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config["max_position_embeddings"]`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end.
        """
        self.layoutxlm.resize_position_embeddings(new_num_position_embeddings)

    def construct(
            self,
            input_ids=None,
            bbox=None,
            image=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            labels=None,
    ):
        outputs = self.layoutxlm(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        seq_length = input_ids.shape[1]
        # sequence out and image out
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        hidden_states_list = []
        for idx in range(self.layoutxlm.num_hidden_layers):
            hidden_states_list.append((f"hidden_states_{idx}", outputs[2][f"{idx}_data"]))

        hidden_states = tuple(hidden_states_list)
        if self.training:
            outputs = (logits, hidden_states)
        else:
            outputs = (logits,)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = (
                        attention_mask.reshape((-1,)) == 1
                )
                active_logits = logits.reshape((-1, self.num_classes))[active_loss]
                active_labels = labels.reshape(
                    (
                        -1,
                    )
                )[active_loss]
                active_labels = active_labels.astype(ms.float32)
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.reshape((-1, self.num_classes)),
                    labels.reshape(
                        (
                            -1,
                        )
                    ),
                )
            outputs = (loss,) + outputs
        return outputs


class BiaffineAttention(nn.Cell):
    """Implements a biaffine attention operator for binary relation classification."""

    def __init__(self, in_features, out_features):
        super(BiaffineAttention, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.bilinear = nn.BiDense(in_features, in_features, out_features, has_bias=False)
        self.linear = nn.Dense(2 * in_features, out_features)

    def construct(self, x_1, x_2):
        return self.bilinear(x_1, x_2) + self.linear(ops.concat((x_1, x_2), axis=-1))


class REDecoder(nn.Cell):
    def __init__(self, hidden_size=768, hidden_dropout_prob=0.1):
        super(REDecoder, self).__init__()
        self.entity_emb = nn.Embedding(3, hidden_size)
        projection = nn.SequentialCell(
            nn.Dense(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=hidden_dropout_prob),
            nn.Dense(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=hidden_dropout_prob),
        )
        self.ffnn_head = copy.deepcopy(projection)
        self.ffnn_tail = copy.deepcopy(projection)
        self.rel_classifier = BiaffineAttention(hidden_size // 2, 2)
        self.loss_fct = nn.CrossEntropyLoss()

    def build_relation(self, relations, entities):
        batch_size, max_seq_len = ops.shape(entities)[:2]
        new_relations = ops.full(
            size=[batch_size, max_seq_len * max_seq_len, 3], fill_value=-1, dtype=relations.dtype
        )
        for b in range(batch_size):
            if entities[b, 0, 0] <= 2:
                entitie_new = ops.full(size=[512, 3], fill_value=-1, dtype=entities.dtype)
                entitie_new[0, :] = 2
                entitie_new[1:3, 0] = 0  # start
                entitie_new[1:3, 1] = 1  # end
                entitie_new[1:3, 2] = 0  # label
                entities[b] = entitie_new
            entitie_label = entities[b, 1: entities[b, 0, 2] + 1, 2]
            all_possible_relations1 = ops.arange(0, entities[b, 0, 2], dtype=entities.dtype)
            all_possible_relations1 = all_possible_relations1[entitie_label == 1]
            all_possible_relations2 = ops.arange(0, entities[b, 0, 2], dtype=entities.dtype)
            all_possible_relations2 = all_possible_relations2[entitie_label == 2]

            all_possible_relations = ops.stack(
                ops.meshgrid(all_possible_relations1, all_possible_relations2), axis=2
            ).reshape((-1, 2))
            if len(all_possible_relations) == 0:
                all_possible_relations = ops.full_like(all_possible_relations, fill_value=-1, dtype=entities.dtype)
                all_possible_relations[0, 0] = 0
                all_possible_relations[0, 1] = 1

            relation_head = relations[b, 1: relations[b, 0, 0] + 1, 0]
            relation_tail = relations[b, 1: relations[b, 0, 1] + 1, 1]
            positive_relations = ops.stack([relation_head, relation_tail], axis=1)

            all_possible_relations_repeat = all_possible_relations.unsqueeze(dim=1).tile(
                (1, len(positive_relations), 1)
            )
            positive_relations_repeat = positive_relations.unsqueeze(dim=0).tile((len(all_possible_relations), 1, 1))
            mask = ops.all(all_possible_relations_repeat == positive_relations_repeat, axis=2)
            negative_mask = ops.any(mask, axis=1) is False
            negative_relations = all_possible_relations[negative_mask]

            positive_mask = ops.any(mask, axis=0) is True
            positive_relations = positive_relations[positive_mask]
            if negative_mask.sum() > 0:
                reordered_relations = ops.concat([positive_relations, negative_relations])
            else:
                reordered_relations = positive_relations

            relation_per_doc_label = ops.zeros((len(reordered_relations), 1), dtype=reordered_relations.dtype)
            relation_per_doc_label[: len(positive_relations)] = 1
            relation_per_doc = ops.concat([reordered_relations, relation_per_doc_label], axis=1)
            assert len(relation_per_doc[:, 0]) != 0
            new_relations[b, 0] = ms.Tensor(ops.shape(relation_per_doc)[0], dtype=new_relations.dtype)
            new_relations[b, 1: len(relation_per_doc) + 1] = relation_per_doc
            # new_relations.append(relation_per_doc)
        return new_relations, entities

    def get_predicted_relations(self, logits, relations, entities):
        pred_relations = []
        for i, pred_label in enumerate(logits.argmax(-1)):
            if pred_label != 1:
                continue
            rel = ops.full(size=[7, 2], fill_value=-1, dtype=relations.dtype)
            rel[0, 0] = relations[:, 0][i]
            rel[1, 0] = entities[:, 0][relations[:, 0][i] + 1]
            rel[1, 1] = entities[:, 1][relations[:, 0][i] + 1]
            rel[2, 0] = entities[:, 2][relations[:, 0][i] + 1]
            rel[3, 0] = relations[:, 1][i]
            rel[4, 0] = entities[:, 0][relations[:, 1][i] + 1]
            rel[4, 1] = entities[:, 1][relations[:, 1][i] + 1]
            rel[5, 0] = entities[:, 2][relations[:, 1][i] + 1]
            rel[6, 0] = 1
            pred_relations.append(rel)
        return pred_relations

    def construct(self, hidden_states, entities, relations):
        batch_size, max_length, _ = ops.shape(entities)
        relations, entities = self.build_relation(relations, entities)
        loss = 0
        all_pred_relations = ops.full(
            size=[batch_size, max_length * max_length, 7, 2], fill_value=-1, dtype=entities.dtype
        )
        for b in range(batch_size):
            relation = relations[b, 1: relations[b, 0, 0] + 1]
            head_entities = relation[:, 0]
            tail_entities = relation[:, 1]
            relation_labels = relation[:, 2]
            entities_start_index = ms.Tensor(entities[b, 1: entities[b, 0, 0] + 1, 0])
            entities_labels = ms.Tensor(entities[b, 1: entities[b, 0, 2] + 1, 2])
            head_index = entities_start_index[head_entities]
            head_label = entities_labels[head_entities]
            head_label_repr = self.entity_emb(head_label)

            tail_index = entities_start_index[tail_entities]
            tail_label = entities_labels[tail_entities]
            tail_label_repr = self.entity_emb(tail_label)

            tmp_hidden_states = hidden_states[b][head_index]
            if len(tmp_hidden_states.shape) == 1:
                tmp_hidden_states = ops.unsqueeze(tmp_hidden_states, dim=0)
            head_repr = ops.concat((tmp_hidden_states, head_label_repr), axis=-1)

            tmp_hidden_states = hidden_states[b][tail_index]
            if len(tmp_hidden_states.shape) == 1:
                tmp_hidden_states = ops.unsqueeze(tmp_hidden_states, dim=0)
            tail_repr = ops.concat((tmp_hidden_states, tail_label_repr), axis=-1)

            heads = self.ffnn_head(head_repr)
            tails = self.ffnn_tail(tail_repr)
            logits = self.rel_classifier(heads, tails)
            loss += self.loss_fct(logits, relation_labels)
            pred_relations = self.get_predicted_relations(logits, relation, entities[b])
            if len(pred_relations) > 0:
                pred_relations = ops.stack(pred_relations)
                all_pred_relations[b, 0, :, :] = ms.Tensor(ops.shape(pred_relations)[0], dtype=all_pred_relations.dtype)
                all_pred_relations[b, 1: len(pred_relations) + 1, :, :] = pred_relations
        return loss, all_pred_relations


class LayoutXLMForRelationExtraction(nn.Cell):
    def __init__(self, layoutxlm, hidden_size=768, hidden_dropout_prob=0.1, dropout=None):
        super(LayoutXLMForRelationExtraction, self).__init__()
        if isinstance(layoutxlm, dict):
            self.layoutxlm = LayoutXLMModel(**layoutxlm)
        else:
            self.layoutxlm = layoutxlm

        dropout_prob = dropout if dropout is not None else self.layoutxlm.config.hidden_dropout_prob

        self.extractor = REDecoder(hidden_size, hidden_dropout_prob)

        self.dropout = nn.Dropout(p=dropout_prob)

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config["max_position_embeddings"]`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end.
        """
        self.layoutxlm.resize_position_embeddings(new_num_position_embeddings)

    def construct(
            self,
            input_ids,
            bbox,
            image=None,
            attention_mask=None,
            entities=None,
            relations=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            labels=None,
    ):
        outputs = self.layoutxlm(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        seq_length = input_ids.shape[1]
        sequence_output = outputs[0][:, :seq_length]

        sequence_output = self.dropout(sequence_output)
        loss, pred_relations = self.extractor(sequence_output, entities, relations)
        hidden_states = [outputs[2][f"{idx}_data"] for idx in range(self.layoutxlm.num_hidden_layers)]
        hidden_states = ops.stack(hidden_states, axis=1)

        res = dict(loss=loss, pred_relations=pred_relations, hidden_states=hidden_states)
        return res


class NLPBaseModel(nn.Cell):
    def __init__(self,
                 base_model_class,
                 model_class,
                 mode="base",
                 type="ser",
                 pretrained=True,
                 checkpoints=None,
                 **kwargs):
        super(NLPBaseModel, self).__init__()
        pretrained_model_dict = {
            LayoutXLMModel: {
                "base": "./layoutxlm-base-e5255349.ckpt",  # TODO:download link
            }
        }
        if checkpoints is not None:  # load the trained model
            config = LayoutXLMPretrainedConfig()
            params = load_checkpoint(checkpoints)
            model = model_class(config)
            load_param_into_net(model, params)
            self.model = model
        else:  # load the pretrained-model
            pretrained_model_link = pretrained_model_dict[base_model_class][
                mode]
            config = LayoutXLMPretrainedConfig()
            base_model = base_model_class(config)
            if pretrained is True:
                params = load_checkpoint(pretrained_model_link)
                load_param_into_net(base_model, params)
            if type == "ser":
                self.model = model_class(
                    base_model, num_classes=kwargs["num_classes"], dropout=None)
            else:
                self.model = model_class(base_model, dropout=None)
        self.out_channels = 1
        self.use_visual_backbone = True


@register_backbone_class
class LayoutXLMForSer(NLPBaseModel):
    def __init__(self,
                 num_classes,
                 pretrained=True,
                 checkpoints=None,
                 mode="base",
                 **kwargs):
        super(LayoutXLMForSer, self).__init__(
            LayoutXLMModel,
            LayoutXLMForTokenClassification,
            mode,
            "ser",
            pretrained,
            checkpoints,
            num_classes=num_classes)
        if hasattr(self.model.layoutxlm, "use_visual_backbone"
                   ) and self.model.layoutxlm.use_visual_backbone is False:
            self.use_visual_backbone = False

    def construct(self, x):
        if self.use_visual_backbone is True:
            image = x[4]
        else:
            image = None
        x = self.model(
            input_ids=x[0],
            bbox=x[1],
            attention_mask=x[2],
            token_type_ids=x[3],
            image=image,
            position_ids=None,
            head_mask=None,
            labels=None)
        if self.training:
            res = {"backbone_out": x[0]}
            return res
        else:
            return x


@register_backbone_class
class LayoutXLMForRe(NLPBaseModel):
    def __init__(self, pretrained=True, checkpoints=None, mode="base",
                 **kwargs):
        super(LayoutXLMForRe, self).__init__(
            LayoutXLMModel, LayoutXLMForRelationExtraction, mode, "re",
            pretrained, checkpoints)
        if hasattr(self.model.layoutxlm, "use_visual_backbone"
                   ) and self.model.layoutxlm.use_visual_backbone is False:
            self.use_visual_backbone = False

    def construct(self, x):
        if self.use_visual_backbone is True:
            image = x[4]
            entities = x[5]
            relations = x[6]
        else:
            image = None
            entities = x[4]
            relations = x[5]
        x = self.model(
            input_ids=x[0],
            bbox=x[1],
            attention_mask=x[2],
            token_type_ids=x[3],
            image=image,
            position_ids=None,
            head_mask=None,
            labels=None,
            entities=entities,
            relations=relations)
        return x


@register_backbone
def layoutxlm_for_re(pretrained: bool = True, **kwargs) -> LayoutXLMForRe:
    """
    A predefined ResNet-18 for Text Detection.

    Args:
        pretrained: whether to load weights pretrained on ImageNet. Default: True.
        **kwargs: additional parameters to pass to ResNet.

    Returns:
        DetResNet: ResNet model.
    """
    model = LayoutXLMForRe(pretrained, **kwargs)
    return model


@register_backbone
def layoutxlm_for_ser(pretrained: bool = True, **kwargs) -> LayoutXLMForSer:
    model = LayoutXLMForSer(pretrained=pretrained, **kwargs)
    return model


def get_layoutxlm_model_params():
    config = LayoutXLMPretrainedConfig()
    model = LayoutXLMModel(config)
    params_dict = model.parameters_dict()
    params_str = ""
    for key, value in params_dict.items():
        params_str += key
        params_str += "\n"
    with open("param.txt", "w") as f:
        f.write(params_str)


if __name__ == "__main__":
    from test_utils import test_layoutxlm_model, test_token_classification

    # model = LayoutXLMForSer(pretrained=True, checkpoints=None, num_classes=7)
    # model = LayoutXLMForRe(pretrained=True, checkpoints=None)

    config = LayoutXLMPretrainedConfig()
    test_case = "LayoutXLM"

    if test_case == "LayoutXLM":
        model = LayoutXLMModel(config)
        test_layoutxlm_model(model)
    elif test_case == "LayoutXLM-pretrained":
        model = LayoutXLMModel(config)
        params = load_checkpoint("./layoutxlm-base.ckpt")
        load_param_into_net(model, params)
        test_layoutxlm_model(model)
    elif test_case == "TokenClassification":
        config.num_labels = 3
        model = LayoutXLMForTokenClassification(config)
        test_token_classification(model)
    elif test_case == "LayoutXLMSer":
        pass
