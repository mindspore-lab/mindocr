import collections

import numpy as np

from mindspore import Parameter, Tensor, nn, ops
from mindspore.common import dtype as mstype

from mindocr.models.backbones._registry import register_backbone, register_backbone_class

from ..transformer_common.layer import (
    LayoutXLMAttention,
    LayoutXLMEmbeddings,
    LayoutXLMEncoder,
    LayoutXLMLayer,
    LayoutXLMSelfAttention,
    finfo,
)
from .configuration import LayoutLMv3PretrainedConfig


class LayoutLMv3PatchEmbeddings(nn.Cell):
    """
    LayoutLMv3 image (patch) embeddings. This class also automatically interpolates the position embeddings for varying
    image sizes.
    """

    def __init__(self, config):
        super().__init__()

        image_size = (
            config.input_size
            if isinstance(config.input_size, collections.abc.Iterable)
            else (config.input_size, config.input_size)
        )
        patch_size = (
            config.patch_size
            if isinstance(config.patch_size, collections.abc.Iterable)
            else (config.patch_size, config.patch_size)
        )
        self.patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.proj = nn.Conv2d(
            config.num_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size, has_bias=True
        )

    def construct(self, pixel_values: Tensor, position_embedding: Tensor = None):
        embeddings = self.proj(pixel_values)

        if position_embedding is not None:
            # interpolate the position embedding to the corresponding size
            position_embedding = position_embedding.view(1, self.patch_shape[0], self.patch_shape[1], -1)
            position_embedding = position_embedding.transpose(0, 3, 1, 2)
            patch_height, patch_width = embeddings.shape[2], embeddings.shape[3]
            position_embedding = ops.interpolate(position_embedding, size=(patch_height, patch_width), mode="bicubic")
            embeddings = embeddings + position_embedding

        embeddings = embeddings.flatten(start_dim=2).transpose(0, 2, 1)
        return embeddings


class LayoutLMv3TextEmbeddings(LayoutXLMEmbeddings):
    """
    LayoutLMv3 text embeddings. Same as `RobertaEmbeddings` but with added spatial (layout) embeddings.
    """

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def create_position_ids_from_input_ids(self, input_ids: Tensor, padding_idx):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).astype(mstype.int32)
        incremental_indices = (ops.cumsum(mask, axis=1)) * mask
        return incremental_indices.astype(mstype.int64) + padding_idx

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = Tensor(np.arange(self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=np.int64))
        return position_ids.unsqueeze(0).broadcast_to(input_shape)

    def construct(
        self,
        input_ids=None,
        bbox=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, dtype=mstype.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        spatial_position_embeddings = self._cal_spatial_position_embeddings(bbox)

        embeddings = embeddings + spatial_position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LayoutLMv3SelfAttention(LayoutXLMSelfAttention):
    def __init__(self, config):
        super().__init__(config)

    def cogview_attention(self, attention_scores: Tensor, alpha=32):
        """
        https://arxiv.org/abs/2105.13290 Section 2.4 Stabilization of training: Precision Bottleneck Relaxation
        (PB-Relax). A replacement of the original nn.Softmax(dim=-1)(attention_scores). Seems the new attention_probs
        will result in a slower speed and a little bias. Can use allclose(standard_attention_probs,
        cogview_attention_probs, atol=1e-08) for comparison. The smaller atol (e.g., 1e-08), the better.
        """
        scaled_attention_scores = attention_scores / alpha
        max_value = scaled_attention_scores.max(axis=-1).unsqueeze(-1)
        new_attention_scores = (scaled_attention_scores - max_value) * alpha
        return nn.Softmax(axis=-1)(new_attention_scores)

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

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # The attention scores QT K/√d could be significantly larger than input elements, and result in overflow.
        # Changing the computational order into QT(K/√d) alleviates the problem. (https://arxiv.org/pdf/2105.13290.pdf)
        attention_scores = ops.matmul(query_layer / self.attention_head_size_sqrt, key_layer.transpose(0, 1, 3, 2))
        if self.has_relative_attention_bias and self.has_spatial_attention_bias:
            attention_scores += (rel_pos + rel_2d_pos) / self.attention_head_size_sqrt
        elif self.has_relative_attention_bias:
            attention_scores += rel_pos / self.attention_head_size_sqrt

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask.astype(self.dense_dtype)

        # Normalize the attention scores to probabilities.
        # Use the trick of the CogView paper to stablize training
        attention_probs = self.cogview_attention(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ops.matmul(attention_probs, value_layer)

        context_layer = context_layer.transpose(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class LayoutLMv3Attention(LayoutXLMAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self_attention = LayoutLMv3SelfAttention(config)


class LayoutLMv3Layer(LayoutXLMLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = LayoutLMv3Attention(config)


class LayoutLMv3Encoder(LayoutXLMEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.CellList([LayoutLMv3Layer(config) for _ in range(config.num_hidden_layers)])


@register_backbone_class
class LayoutLMv3Model(nn.Cell):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias
        self.patch_size = config.patch_size
        self.float32_min = finfo(mstype.float32)
        self.float16_min = finfo(mstype.float16)
        self.out_channels = 1
        self.use_visual_backbone = True

        if config.text_embed:
            self.embeddings = LayoutLMv3TextEmbeddings(config)

        if config.visual_embed:
            # use the default pre-training parameters for fine-tuning (e.g., input_size)
            # when the input_size is larger in fine-tuning, we will interpolate the position embeddings in forward
            self.patch_embed = LayoutLMv3PatchEmbeddings(config)

            size = int(config.input_size / config.patch_size)
            self.cls_token = Parameter(ops.zeros((1, 1, config.hidden_size)))
            self.pos_embed = Parameter(ops.zeros((1, size * size + 1, config.hidden_size)))
            self.pos_drop = nn.Dropout(p=0.0)

            self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
            self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

            if config.has_relative_attention_bias or config.has_spatial_attention_bias:
                self.init_visual_bbox(image_size=(size, size))

            self.norm = nn.LayerNorm((config.hidden_size,), epsilon=1e-6)

        self.encoder = LayoutLMv3Encoder(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def init_visual_bbox(self, image_size=(14, 14), max_len=1000):
        """
        Create the bounding boxes for the visual (patch) tokens.
        """
        visual_bbox_x = ops.truncate_div(Tensor(np.arange(0, max_len * (image_size[1] + 1), max_len)), image_size[1])
        visual_bbox_y = ops.truncate_div(Tensor(np.arange(0, max_len * (image_size[0] + 1), max_len)), image_size[0])
        visual_bbox = ops.stack(
            [
                visual_bbox_x[:-1].broadcast_to((image_size[0], -1)),
                visual_bbox_y[:-1].broadcast_to((image_size[1], -1)).transpose(0, 1),
                visual_bbox_x[1:].broadcast_to((image_size[0], -1)),
                visual_bbox_y[1:].broadcast_to((image_size[1], -1)).transpose(0, 1),
            ],
            axis=-1,
        ).reshape(-1, 4)

        cls_token_box = Tensor([[0 + 1, 0 + 1, max_len - 1, max_len - 1]])
        self.visual_bbox = ops.cat([cls_token_box, visual_bbox], axis=0)

    def calculate_visual_bbox(self, dtype, batch_size):
        final_shape = self.visual_bbox.shape
        visual_bbox = self.visual_bbox.broadcast_to((batch_size, final_shape[0], final_shape[1]))
        visual_bbox = visual_bbox.astype(dtype)
        return visual_bbox

    def visual_embeddings(self, pixel_values):
        embeddings = self.patch_embed(pixel_values)

        # add [CLS] token
        batch_size, seq_len, _ = embeddings.shape
        cls_tokens = self.cls_token.broadcast_to((batch_size, -1, -1))
        embeddings = ops.cat((cls_tokens, embeddings), axis=1)

        # add position embeddings
        if self.pos_embed is not None:
            embeddings = embeddings + self.pos_embed

        embeddings = self.pos_drop(embeddings)
        embeddings = self.norm(embeddings)

        return embeddings

    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape, dtype) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if attention_mask.ndim == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.  # fp16 compatibility
        extended_attention_mask = extended_attention_mask.astype(dtype)

        if dtype == mstype.float32:
            minimum = self.float32_min
        elif dtype == mstype.float16:
            minimum = self.float16_min

        extended_attention_mask = (1.0 - extended_attention_mask) * minimum
        return extended_attention_mask

    def get_head_mask(self, head_mask, num_hidden_layers: int, is_attention_chunked: bool = False):
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask: Tensor, num_hidden_layers: int):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.ndim == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.broadcast_to((self.num_hidden_layers, -1, -1, -1, -1))
        elif head_mask.ndim == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        if head_mask.ndim != 5:
            raise ValueError(f"head_mask.dim != 5, instead {head_mask.ndim}")
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def construct(
        self,
        input_ids=None,  # input_ids
        bbox=None,  # b_box
        attention_mask=None,  # attention_mask
        token_type_ids=None,  # token_type_ids
        pixel_values=None,  # image
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Constructs the LayoutLMv3 model according to the input provided.

        Args:
            input_ids (Tensor, optional): Tensor containing the token IDs of the input text sequence.
            bbox (Tensor, optional): Tensor containing the bounding box information of the input text sequence.
            attention_mask (Tensor, optional): Tensor containing the attention mask for the input sequence.
            token_type_ids (Tensor, optional): Tensor containing the token type IDs to distinguish different sequences.
            pixel_values (Tensor, optional): Tensor containing the pixel values of the input image.
            position_ids (Tensor, optional): Tensor containing the position IDs indicating the position of tokens.
            head_mask (Tensor, optional): Mask to control which heads of the attention mechanism should be used.
            inputs_embeds (Tensor, optional): Pre-computed embeddings for the input tokens.
            output_attentions (bool, optional): Whether to return attention weights.
            output_hidden_states (bool, optional): Whether to return hidden states.
            return_dict (bool, optional): Whether to return a dictionary or a tuple of outputs.

        Returns:
            Tensor or Tuple[Tensor]: Depending on the configuration, returns either a tensor or a tuple
            containing the output sequence and additional outputs such as hidden states and attention weights.
        """
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else False
        seq_length = None
        input_shape = None
        if input_ids is not None:
            input_shape = input_ids.shape
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            batch_size, seq_length = input_shape
        elif pixel_values is not None:
            batch_size = len(pixel_values)
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or pixel_values")
        embedding_output = None

        if input_ids is not None or inputs_embeds is not None:
            if attention_mask is None:
                attention_mask = ops.ones(((batch_size, seq_length)))
            if token_type_ids is None:
                token_type_ids = ops.zeros(input_shape, dtype=mstype.int64)
            if bbox is None:
                bbox = ops.zeros(tuple(list(input_shape) + [4]), dtype=mstype.int64)

            # ocr information text embeddings
            embedding_output = self.embeddings(
                input_ids=input_ids,
                bbox=bbox,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )
        final_bbox = final_position_ids = None
        if pixel_values is not None:
            visual_embeddings = self.visual_embeddings(pixel_values)
            visual_embeddings_shape = visual_embeddings.shape
            visual_attention_mask = ops.ones((batch_size, visual_embeddings_shape[1]), dtype=mstype.int64)
            if attention_mask is not None:
                attention_mask = ops.cat([attention_mask, visual_attention_mask.astype(attention_mask.dtype)], axis=1)
            else:
                attention_mask = visual_attention_mask

            if self.has_relative_attention_bias or self.has_spatial_attention_bias:
                if self.has_spatial_attention_bias:
                    visual_bbox = self.calculate_visual_bbox(dtype=mstype.int64, batch_size=batch_size)
                    if bbox is not None:
                        final_bbox = ops.cat([bbox, visual_bbox], axis=1)
                    else:
                        final_bbox = visual_bbox

                visual_embeddings_shape = visual_embeddings.shape
                visual_position_ids = ops.arange(0, visual_embeddings_shape[1], dtype=mstype.int64).broadcast_to(
                    (batch_size, visual_embeddings_shape[1])
                )
                if input_ids is not None or inputs_embeds is not None:
                    position_ids = ops.arange(0, input_shape[1], dtype=mstype.int64).unsqueeze(0)
                    position_ids = position_ids.broadcast_to(input_shape)
                    final_position_ids = ops.cat([position_ids, visual_position_ids], axis=1)
                else:
                    final_position_ids = visual_position_ids

            if input_ids is not None or inputs_embeds is not None:
                embedding_output = ops.cat([embedding_output, visual_embeddings], axis=1)
            else:
                embedding_output = visual_embeddings

            embedding_output = self.LayerNorm(embedding_output)
            embedding_output = self.dropout(embedding_output)
        elif self.has_relative_attention_bias or self.has_spatial_attention_bias:
            if self.has_spatial_attention_bias:
                final_bbox = bbox
            if self.has_relative_attention_bias:
                position_ids = self.embeddings.position_ids[:, : input_shape[1]]
                position_ids = position_ids.expand_as(input_ids)
                final_position_ids = position_ids

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, None, embedding_output.dtype)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            bbox=final_bbox,
            position_ids=final_position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = encoder_outputs[0]

        return (sequence_output,) + encoder_outputs[1:]


@register_backbone
def layoutlmv3(**kwargs):
    pretrained_config = LayoutLMv3PretrainedConfig()
    model = LayoutLMv3Model(pretrained_config)
    return model
