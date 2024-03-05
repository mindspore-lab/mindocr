import numpy as np

import mindspore as ms
from mindspore import ops
from mindspore import nn

from mindocr.models.third_party.mindformers.research.qwen.qwen_model import QwenForCausalLM, QwenModel
from mindocr.models.third_party.mindformers.research.qwen.qwen_config import QwenConfig
from mindocr.models.third_party.mindformers.mindformers.models import ImageEncoderConfig

from mindocr.models.llm.vary_clip_model import build_model
from mindocr.models.llm.vary_sam_model import SAMEncoder


class VaryConfig(QwenConfig):
    model_type = "vary"


class VaryQwenModel(QwenModel):
    def __init__(self, config: QwenConfig):
        super(VaryQwenModel, self).__init__(config)
        sam_config_dict = dict(img_size=1024,  # img_size in ImageEncoderViT
                               patch_size=16,  # patch_size in ImageEncoderViT
                               in_chans=3,  # in_chans in ImageEncoderViT
                               embed_dim=768,  # encoder_embed_dim in build_sam_vit_b
                               depth=12,  # encoder_depth in build_sam_vit_b
                               num_heads=12,  # encoder_num_heads in build_sam_vit_b
                               mlp_ratio=4,  # mlp_ratio in ImageEncoderViT
                               out_chans=256,  # out_chans in ImageEncoderViT
                               qkv_bias=True,  # qkv_bias in ImageEncoderViT
                               layer_norm_eps=1.e-6,  # refer to Vary-master\vary\model\vision_encoder\sam.py
                               use_abs_pos=True,  # use_abs_pos in ImageEncoderViT
                               use_rel_pos=True,  # use_rel_pos in ImageEncoderViT
                               window_size=14,
                               global_attn_indexes=[2, 5, 8, 11],  # encoder_global_attn_indexes in build_sam_vit_b
                               compute_dtype="float32",
                               layernorm_compute_type="float32",
                               softmax_compute_type="float32",
                               param_init_type="float32")
        sam_config = ImageEncoderConfig(**sam_config_dict)
        self.vision_tower_high = SAMEncoder(sam_config)
        self.vision_tower_high.to_float(ms.float16)

        self.vision_tower = build_model()
        self.vision_tower.to_float(ms.float16)

        self.mm_projector = nn.Dense(1024, 1024).to_float(ms.float16)
        self.mm_projector_vary = nn.Dense(1024, 1024).to_float(ms.float16)

        self.vision_select_layer = getattr(self.config, "vision_select_layer", -1)

        self.image_start_token_pos = 22
        self.num_patches = 256

    def construct(
            self,
            input_ids,
            init_reset=True,
            batch_valid_length=None,
            batch_index=None,
            zactivate_len=None,
            image=None,
            image_high=None,
    ):
        # 1. wte
        input_shape = input_ids.shape
        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = self.wte(input_ids)

        if input_shape[1] > 1:
            img_ts = image_high
            sam_out = self.vision_tower_high(img_ts)
            sam_out = self.mm_projector_vary(sam_out)

            img_ts = image
            clip_out = self.vision_tower(img_ts)
            clip_out = self.mm_projector(clip_out)

            image_features = ops.concat((clip_out, sam_out), -1)

            new_input_embeds = []
            num_patches = self.num_patches
            image_start_token_pos = self.image_start_token_pos
            for i in range(input_shape[0]):
                cur_input_embeds = inputs_embeds[i]
                per_cur_image_features = image_features[i]
                cur_input_embeds = ops.cat(
                    (
                        cur_input_embeds[:image_start_token_pos + 1],
                        per_cur_image_features,
                        cur_input_embeds[image_start_token_pos + num_patches + 1:]
                    ),
                    axis=0
                )

                new_input_embeds.append(cur_input_embeds)

            hidden_states = ops.stack(new_input_embeds, axis=0)
        else:
            hidden_states = inputs_embeds

        # 2. drop
        hidden_states = self.drop(hidden_states)

        # 2. rotary_emb
        bs, seq_len = self.shape(input_ids)
        if not self.use_past:
            freqs_cis = self.freqs_mgr()
            mask = self.casual_mask(input_ids)  # mask: [bs, seq, seq]
            mask = self.casual_mask.post_process(mask)
            kvcache_inputs = None
        else:
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr(seq_len)
                mask = self.casual_mask(input_ids)  # mask: [bs, seq, seq]
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length, bs)
                if self.is_dynamic and self.is_flexible_shape and not self.use_kvcache_op:
                    mask = self.casual_mask.increment_slice(self.kvcache_preprocess.range,
                                                            self.kvcache_preprocess.max_cache_length // bs,
                                                            batch_valid_length,
                                                            zactivate_len)
                else:
                    mask = self.casual_mask.increment(self.kvcache_preprocess.range, batch_valid_length, zactivate_len)
            mask = self.casual_mask.post_process(mask)

            kvcache_inputs = self.kvcache_preprocess(bs, batch_valid_length, batch_index, zactivate_len)

        # 4. hidden_states
        for i in range(self.num_hidden_layers):
            hidden_states = self.layers[i](hidden_states, freqs_cis, mask, kvcache_inputs=kvcache_inputs)

        # 5. ln_f
        hidden_states = self.ln_f(hidden_states)

        return hidden_states


class VaryQwenForCausalLM(QwenForCausalLM):
    def __init__(self, config):
        super(VaryQwenForCausalLM, self).__init__(config)
        self.transformer = VaryQwenModel(config=config)
        self.load_checkpoint(config)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": ms.Tensor(input_ids, ms.int32),
            "image": ms.Tensor(kwargs["image"], ms.float16),
            "image_high": ms.Tensor(kwargs["image_high"], ms.float16)
        }

    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  image=None, image_high=None):
        """construct"""
        bsz, seqlen = input_ids.shape
        if self.use_past:
            if not isinstance(batch_valid_length, ms.Tensor):
                batch_valid_length = self.ones((bsz,), ms.int32)
        if self.training:
            tokens = self.slice(input_ids, (0, 0), (bsz, seqlen - 1), (1, 1))
        else:
            tokens = input_ids

        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))
        if not self.is_first_iteration:
            batch_valid_length = self.sub_batch_valid_len(batch_valid_length, 1)

        output = self.transformer(tokens, init_reset=init_reset, batch_valid_length=batch_valid_length,
                                  batch_index=batch_index, zactivate_len=zactivate_len,
                                  image=image, image_high=image_high)
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
            output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        logits = self.lm_head(output)

        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), ms.float32)
        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bsz, seqlen), (1, 1))
        else:
            if labels.ndim > 1:
                if self.training:
                    labels = self.slice(labels, (0, 1), (bsz, seqlen), (1, 1))
                label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), ms.float32)
                input_mask = self.mul(input_mask, label_mask)

        if not self.training:
            if not pre_gather:
                logits = self.reshape(logits, (bsz, seqlen, -1))
            logits = self.cast(logits, ms.float32)
            # makes cast effective to avoid allgather issue in Mindspore1.10
            input_mask = self.add(input_mask, 1)
            return logits, tokens, input_mask

        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        logits = self.cast(logits, ms.float32)
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss
