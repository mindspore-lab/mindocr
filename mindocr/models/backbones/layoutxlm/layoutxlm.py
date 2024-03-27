import numpy as np

from mindspore import Parameter, Tensor, nn, ops, set_context
from mindspore.common import dtype as mstype

from .._registry import register_backbone, register_backbone_class
from ..mindcv_models.utils import load_pretrained
from ..transformer_common.layer import LayoutXLMEmbeddings, LayoutXLMEncoder, LayoutXLMPooler
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
        url="https://download-mindspore.osinfra.cn/toolkits/mindocr/layoutxlm/layoutxlm_base_uncased-00b703e2.ckpt",
        use_visual_backbone=True,
    ),
    "vi-layoutxlm": _cfg(
        url="https://download-mindspore.osinfra.cn/toolkits/mindocr/vi-layoutxlm/vi_layoutxlm_uncased-5ae4d2b8.ckpt",
        use_visual_backbone=False,
    ),
}


class VisualBackbone(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.cfg = read_config()
        self.backbone = build_resnet_fpn_backbone(self.cfg)

        if len(self.cfg.MODEL.PIXEL_MEAN) != len(self.cfg.MODEL.PIXEL_STD):
            raise ValueError("cfg.model.pixel_mean is not equal with cfg.model.pixel_std.")
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)

        self.pixel_mean = Parameter(
            Tensor(self.cfg.MODEL.PIXEL_MEAN).reshape((num_channels, 1, 1)),
            name="pixel_mean",
            requires_grad=False,
        )
        self.pixel_std = Parameter(
            Tensor(self.cfg.MODEL.PIXEL_STD).reshape((num_channels, 1, 1)),
            name="pixel_std",
            requires_grad=False,
        )

        self.out_feature_key = "p2"
        self.pool_shape = tuple(config.image_feature_pool_shape[:2])  # (7,7)
        if len(config.image_feature_pool_shape) == 2:
            config.image_feature_pool_shape.append(self.backbone.output_shape()[self.out_feature_key].channels)

        input_shape = (224, 224)
        outsize = config.image_feature_pool_shape[0]  # (7,7)
        insize = (input_shape[0] + 4 - 1) // 4
        shape_info = self.backbone.output_shape()[self.out_feature_key]
        channels = shape_info.channels
        stride = insize // outsize
        kernel = insize - (outsize - 1) * stride

        self.weight = Tensor(np.ones([channels, 1, kernel, kernel]), dtype=mstype.float32) / (kernel * kernel)
        self.conv2d = ops.Conv2D(channels, kernel, stride=stride, group=channels)

    def pool(self, features):
        """
        Custom AvgPool2d
        """
        features = self.conv2d(features, self.weight)
        return features

    def freeze(self):
        """
        Freeze parameters
        """
        for param in self.trainable_params():
            param.requires_grad = False

    def construct(self, images):
        images_input = (images - self.pixel_mean) / self.pixel_std
        features = self.backbone(images_input)
        for item in features:
            if item[0] == self.out_feature_key:
                features = item[1]
        features = self.pool(features)
        return features.flatten(start_dim=2).transpose(0, 2, 1)


@register_backbone_class
class LayoutXLMModel(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.has_visual_segment_embedding = config.has_visual_segment_embedding
        self.embeddings = LayoutXLMEmbeddings(config)
        self.use_visual_backbone = config.use_visual_backbone
        self.use_float16 = config.use_float16
        self.dense_dtype = mstype.float32
        if self.use_float16 is True:
            self.dense_dtype = mstype.float16

        if self.use_visual_backbone is True:
            set_context(jit_syntax_level=0)
            self.visual = VisualBackbone(config)
            self.visual.freeze()
            self.visual_proj = nn.Dense(config.image_feature_pool_shape[-1], config.hidden_size).to_float(
                self.dense_dtype
            )
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
        self.out_channels = 1

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
        use_image_info = self.use_visual_backbone and image is not None
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(bbox)
        if use_image_info:
            visual_embeddings = self.visual_proj(self.visual(image.astype(mstype.float32)))
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
        num_position_embeds_diff = new_num_position_embeddings - self.max_position_embeddings

        # no resizing needs to be done if the length stays the same
        if num_position_embeds_diff == 0:
            return

        self.max_position_embeddings = new_num_position_embeddings

        old_position_embeddings_weight = self.embeddings.position_embeddings.embedding_table

        self.embeddings.position_embeddings = nn.Embedding(self.max_position_embeddings, self.hidden_size)

        if num_position_embeds_diff > 0:
            self.embeddings.position_embeddings.embedding_table[
                :-num_position_embeds_diff
            ] = old_position_embeddings_weight
        else:
            self.embeddings.position_embeddings.embedding_table = old_position_embeddings_weight[
                :num_position_embeds_diff
            ]

    def _calc_visual_bbox(self, image_feature_pool_shape, bbox, visual_shape):
        x_size = image_feature_pool_shape[1]
        y_size = image_feature_pool_shape[0]
        visual_bbox_x = Tensor(np.arange(0, 1000 * (x_size + 1), 1000) // x_size, dtype=mstype.int64)
        visual_bbox_y = Tensor(np.arange(0, 1000 * (y_size + 1), 1000) // y_size, dtype=mstype.int64)
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
        attention_mask=None,
        token_type_ids=None,
        image=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        input_shape = self._get_input_shape(input_ids, inputs_embeds)
        visual_shape = list(input_shape)
        visual_shape[1] = self.image_feature_pool_shape_size
        visual_bbox = self._calc_visual_bbox(self.image_feature_pool_shape, bbox, visual_shape)

        final_bbox = ops.concat([bbox, visual_bbox], axis=1)
        if attention_mask is None:
            attention_mask = ops.ones(input_shape)

        if self.use_visual_backbone is True:
            visual_attention_mask = ops.ones(visual_shape)
        else:
            visual_attention_mask = ops.zeros(visual_shape)

        attention_mask = attention_mask.astype(visual_attention_mask.dtype)

        final_attention_mask = ops.concat([attention_mask, visual_attention_mask], axis=1)

        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, dtype=mstype.int64)

        if position_ids is None:
            seq_length = input_shape[1]
            position_ids = self.embeddings.position_ids[:, :seq_length]
            position_ids = position_ids.broadcast_to(input_shape)

        visual_position_ids = Tensor(np.arange(0, visual_shape[1])).broadcast_to((input_shape[0], visual_shape[1]))
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
            attention_mask=extended_attention_mask,
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
def layoutxlm(pretrained: bool = True, use_visual_backbone: bool = True, use_float16: bool = False, **kwargs):
    pretrained_config = LayoutXLMPretrainedConfig(use_visual_backbone, use_float16)
    model = LayoutXLMModel(pretrained_config)
    if pretrained:
        if use_visual_backbone is True:
            default_cfg = default_cfgs["layoutxlm"]
        else:
            default_cfg = default_cfgs["vi-layoutxlm"]
        load_pretrained(model, default_cfg)
    return model
