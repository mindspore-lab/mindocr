from .configuration import (
    LAYOUTXLM_PRETRAINED_INIT_CONFIGURATION,
    LAYOUTXLM_PRETRAINED_RESOURCE_FILES_MAP,
    LayoutXLMConfig,
)
from .model_utils import PretrainedModel
from .visual_backbone import build_resnet_fpn_backbone, read_config

class LayoutXLMEmbeddings(Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self, config: LayoutXLMConfig):
        super(LayoutXLMEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)

        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids", paddle.arange(config.max_position_embeddings, dtype="int64").expand((1, -1))
        )

    def _cal_spatial_position_embeddings(self, bbox):
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The :obj:`bbox`coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

        spatial_position_embeddings = paddle.concat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            axis=-1,
        )
        return spatial_position_embeddings

    def forward(self, input_ids, bbox=None, token_type_ids=None, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)

            position_ids = seq_length - ones
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

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


class LayoutXLMPretrainedModel(PretrainedModel):

    config_class = LayoutXLMConfig
    pretrained_init_configuration = LAYOUTXLM_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = LAYOUTXLM_PRETRAINED_RESOURCE_FILES_MAP
    base_model_prefix = "layoutxlm"

    def _init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.pretrained_init_configuration["initializer_range"]
                        if "initializer_range" in self.pretrained_init_configuration
                        else 0.02,
                        shape=layer.weight.shape,
                    )
                )


class LayoutXLMEncoder(nn.Layer):
    def __init__(self, config: LayoutXLMConfig):
        super(LayoutXLMEncoder, self).__init__()
        self.config = config
        self.layer = nn.LayerList([LayoutXLMLayer(config) for _ in range(config.num_hidden_layers)])

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_onehot_size = config.rel_pos_bins
            self.rel_pos_bias = nn.Linear(self.rel_pos_onehot_size, config.num_attention_heads, bias_attr=False)

        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_2d_pos_onehot_size = config.rel_2d_pos_bins
            self.rel_pos_x_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias_attr=False)
            self.rel_pos_y_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias_attr=False)

    def _cal_1d_pos_emb(self, hidden_states, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        rel_pos = paddle.nn.functional.one_hot(rel_pos, num_classes=self.rel_pos_onehot_size).astype(
            hidden_states.dtype
        )
        rel_pos = self.rel_pos_bias(rel_pos).transpose([0, 3, 1, 2])
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
        rel_pos_x = F.one_hot(rel_pos_x, num_classes=self.rel_2d_pos_onehot_size).astype(hidden_states.dtype)
        rel_pos_y = F.one_hot(rel_pos_y, num_classes=self.rel_2d_pos_onehot_size).astype(hidden_states.dtype)
        rel_pos_x = self.rel_pos_x_bias(rel_pos_x).transpose([0, 3, 1, 2])
        rel_pos_y = self.rel_pos_y_bias(rel_pos_y).transpose([0, 3, 1, 2])
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def forward(
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

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # gradient_checkpointing is set as False here so we remove some codes here
            hidden_save["input_attention_mask"] = attention_mask
            hidden_save["input_layer_head_mask"] = layer_head_mask
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


class LayoutXLMPooler(Layer):
    def __init__(self, config: LayoutXLMConfig):
        super(LayoutXLMPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.with_pool = config.with_pool

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.with_pool == "tanh":
            pooled_output = self.activation(pooled_output)
        return pooled_output


class VisualBackbone(nn.Layer):
    def __init__(self, config: LayoutXLMConfig):
        super(VisualBackbone, self).__init__()
        self.cfg = read_config()
        self.backbone = build_resnet_fpn_backbone(self.cfg)

        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        self.register_buffer("pixel_mean", paddle.to_tensor(self.cfg.MODEL.PIXEL_MEAN).reshape([num_channels, 1, 1]))
        self.register_buffer("pixel_std", paddle.to_tensor(self.cfg.MODEL.PIXEL_STD).reshape([num_channels, 1, 1]))
        self.out_feature_key = "p2"
        # is_deterministic is disabled here.
        self.pool = nn.AdaptiveAvgPool2D(config.image_feature_pool_shape[:2])
        if len(config.image_feature_pool_shape) == 2:
            config.image_feature_pool_shape.append(self.backbone.output_shape()[self.out_feature_key].channels)
        assert self.backbone.output_shape()[self.out_feature_key].channels == config.image_feature_pool_shape[2]

    def forward(self, images):
        images_input = (paddle.to_tensor(images) - self.pixel_mean) / self.pixel_std
        features = self.backbone(images_input)
        features = features[self.out_feature_key]
        features = self.pool(features).flatten(start_axis=2).transpose([0, 2, 1])
        return features


class LayoutXLMModel(LayoutXLMPretrainedModel):
    """
    The bare LayoutXLM Model outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (`int`):
            Vocabulary size of the XLNet model. Defines the number of different tokens that can
            be represented by the `inputs_ids` passed when calling XLNetModel.
        hidden_size (`int`, optional):
            Dimensionality of the encoder layers and the pooler layer. Defaults to ``768``.
        num_hidden_layers (`int`, optional):
            Number of hidden layers in the Transformer encoder. Defaults to ``12``.
        num_attention_heads (`int`, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to ``12``.
        intermediate_size (`int`, optional):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
            Defaults to ``3072``.
        hidden_act (`str`, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to ``"gelu"``.
        hidden_dropout_prob (`float`, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to ``0.1``.
        attention_probs_dropout_prob (`float`, optional):
            The dropout probability for all fully connected layers in the pooler.
            Defaults to ``0.1``.
        initializer_range (`float`, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            Defaults to ``0.02``.
    """

    def __init__(self, config: LayoutXLMConfig):
        super(LayoutXLMModel, self).__init__(config)
        self.config = config
        self.use_visual_backbone = config.use_visual_backbone
        self.has_visual_segment_embedding = config.has_visual_segment_embedding
        self.embeddings = LayoutXLMEmbeddings(config)

        if self.use_visual_backbone is True:
            self.visual = VisualBackbone(config)
            self.visual.stop_gradient = True
            self.visual_proj = nn.Linear(config.image_feature_pool_shape[-1], config.hidden_size)

        if self.has_visual_segment_embedding:
            self.visual_segment_embedding = self.create_parameter(
                shape=[
                    config.hidden_size,
                ],
                dtype=paddle.float32,
            )
        self.visual_LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.visual_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.encoder = LayoutXLMEncoder(config)
        self.pooler = LayoutXLMPooler(config)

    def _calc_text_embeddings(self, input_ids, bbox, position_ids, token_type_ids):
        words_embeddings = self.embeddings.word_embeddings(input_ids)
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(bbox)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + spatial_position_embeddings + token_type_embeddings
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings

    def _calc_visual_bbox(self, image_feature_pool_shape, bbox, visual_shape):
        visual_bbox_x = (
            paddle.arange(
                0,
                1000 * (image_feature_pool_shape[1] + 1),
                1000,
                dtype=bbox.dtype,
            )
            // image_feature_pool_shape[1]
        )
        visual_bbox_y = (
            paddle.arange(
                0,
                1000 * (image_feature_pool_shape[0] + 1),
                1000,
                dtype=bbox.dtype,
            )
            // image_feature_pool_shape[0]
        )

        expand_shape = image_feature_pool_shape[0:2]
        visual_bbox = paddle.stack(
            [
                visual_bbox_x[:-1].expand(expand_shape),
                visual_bbox_y[:-1].expand(expand_shape[::-1]).transpose([1, 0]),
                visual_bbox_x[1:].expand(expand_shape),
                visual_bbox_y[1:].expand(expand_shape[::-1]).transpose([1, 0]),
            ],
            axis=-1,
        ).reshape([expand_shape[0] * expand_shape[1], paddle.shape(bbox)[-1]])

        visual_bbox = visual_bbox.expand([visual_shape[0], visual_bbox.shape[0], visual_bbox.shape[1]])
        return visual_bbox

    def _calc_img_embeddings(self, image, bbox, position_ids):
        use_image_info = self.use_visual_backbone and image is not None
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(bbox)
        if use_image_info is True:
            visual_embeddings = self.visual_proj(self.visual(image.astype(paddle.float32)))
            embeddings = visual_embeddings + position_embeddings + spatial_position_embeddings
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
        num_position_embeds_diff = new_num_position_embeddings - self.config.max_position_embeddings

        # no resizing needs to be done if the length stays the same
        if num_position_embeds_diff == 0:
            return

        logger.info(f"Setting `config.max_position_embeddings={new_num_position_embeddings}`...")
        self.config.max_position_embeddings = new_num_position_embeddings

        old_position_embeddings_weight = self.embeddings.position_embeddings.weight

        self.embeddings.position_embeddings = nn.Embedding(
            self.config.max_position_embeddings, self.config.hidden_size
        )

        with paddle.no_grad():
            if num_position_embeds_diff > 0:
                self.embeddings.position_embeddings.weight[:-num_position_embeds_diff] = old_position_embeddings_weight
            else:
                self.embeddings.position_embeddings.weight = old_position_embeddings_weight[:num_position_embeds_diff]

    def forward(
        self,
        input_ids=None,
        bbox=None,
        image=None,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        head_mask=None,
        output_hidden_states=False,
        output_attentions=False,
    ):
        input_shape = paddle.shape(input_ids)
        visual_shape = list(input_shape)
        visual_shape[1] = self.config.image_feature_pool_shape[0] * self.config.image_feature_pool_shape[1]
        visual_bbox = self._calc_visual_bbox(self.config.image_feature_pool_shape, bbox, visual_shape)

        final_bbox = paddle.concat([bbox, visual_bbox], axis=1)
        if attention_mask is None:
            attention_mask = paddle.ones(input_shape)

        if self.use_visual_backbone is True:
            visual_attention_mask = paddle.ones(visual_shape)
        else:
            visual_attention_mask = paddle.zeros(visual_shape)

        attention_mask = attention_mask.astype(visual_attention_mask.dtype)

        final_attention_mask = paddle.concat([attention_mask, visual_attention_mask], axis=1)

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)

        if position_ids is None:
            seq_length = input_shape[1]
            position_ids = self.embeddings.position_ids[:, :seq_length]
            position_ids = position_ids.expand(input_shape)

        visual_position_ids = paddle.arange(0, visual_shape[1]).expand([input_shape[0], visual_shape[1]])
        final_position_ids = paddle.concat([position_ids, visual_position_ids], axis=1)

        if bbox is None:
            bbox = paddle.zeros(input_shape + [4])

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
        final_emb = paddle.concat([text_layout_emb, visual_emb], axis=1)

        extended_attention_mask = final_attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        else:
            head_mask = [None] * self.config.num_hidden_layers

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


class LayoutXLMForTokenClassification(LayoutXLMPretrainedModel):
    def __init__(self, config: LayoutXLMConfig):
        super(LayoutXLMForTokenClassification, self).__init__(config)
        self.num_classes = config.num_labels
        self.layoutxlm = LayoutXLMModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_classes)

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

    def forward(
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

        hidden_states = {
            f"hidden_states_{idx}": outputs[2][f"{idx}_data"] for idx in range(self.layoutxlm.config.num_hidden_layers)
        }
        if self.training:
            outputs = (logits, hidden_states)
        else:
            outputs = (logits,)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = (
                    attention_mask.reshape(
                        [
                            -1,
                        ]
                    )
                    == 1
                )
                active_logits = logits.reshape([-1, self.num_classes])[active_loss]
                active_labels = labels.reshape(
                    [
                        -1,
                    ]
                )[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.reshape([-1, self.num_classes]),
                    labels.reshape(
                        [
                            -1,
                        ]
                    ),
                )

            outputs = (loss,) + outputs

        return outputs


class LayoutXLMForRelationExtraction(LayoutXLMPretrainedModel):
    def __init__(self, config: LayoutXLMConfig):
        super(LayoutXLMForRelationExtraction, self).__init__(config)

        self.layoutxlm = LayoutXLMModel(config)

        self.extractor = REDecoder(config.hidden_size, config.hidden_dropout_prob)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def _init_weights(self, layer):
        """Initialize the weights"""
        if isinstance(layer, nn.Linear):
            layer.weight.set_value(paddle.tensor.normal(mean=0.0, std=0.02, shape=layer.weight.shape))
            if layer.bias is not None:
                layer.bias.set_value(paddle.tensor.zeros(shape=layer.bias.shape))
        elif isinstance(layer, nn.Embedding):
            layer.weight.set_value(paddle.tensor.normal(mean=0.0, std=0.02, shape=layer.weight.shape))
            if layer._padding_idx is not None:
                layer.weight[layer._padding_idx].set_value(
                    paddle.tensor.normal(mean=0.0, std=0.02, shape=layer.weight[layer._padding_idx].shape)
                )
        elif isinstance(layer, nn.LayerNorm):
            layer.weight.set_value(paddle.tensor.ones(shape=layer.bias.shape))
            layer.bias.set_value(paddle.tensor.zeros(shape=layer.bias.shape))

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

    def forward(
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
        hidden_states = [outputs[2][f"{idx}_data"] for idx in range(self.layoutxlm.config.num_hidden_layers)]
        hidden_states = paddle.stack(hidden_states, axis=1)

        res = dict(loss=loss, pred_relations=pred_relations, hidden_states=hidden_states)
        return res