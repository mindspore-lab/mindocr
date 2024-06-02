import mindspore as ms
from mindspore import ops

from mindocr.nlp.llm import register_llm
from mindocr.nlp.llm.qwen_model import QwenModel, QwenForCausalLM
from mindocr.nlp.llm.configs import MonkeyConfig
from mindocr.nlp.utils.layers import Linear
from mindocr.nlp.llm.vary_clip_model import VisionTransformer
from mindocr.utils.conversation import Conversation
from mindocr.data.transforms.llm_transform import MonkeyImageProcessor


class MonkeyModel(QwenModel):
    def __init__(self, config):
        super().__init__(config)
        self.image_start_token_pos = 0
        self.num_patches = 1280

        self.visual = VisionTransformer(
            input_resolution=config.visual.get("image_size", 896),  # image_size in transformers
            patch_size=config.visual.get("patch_size", 14),  # patch_size in transformers
            width=config.visual.get("width", 1664),  # hidden_size
            layers=config.visual.get("layers", 48),  # num_hidden_layers
            heads=config.visual.get("heads", 16),  # num_attention_heads
            output_dim=config.visual.get("output_dim", 4096),  # projection_dim in transformers
            vision_select_layer=-2,
            param_init_type=config.param_init_type,
            ln_param_init_type=config.ln_param_init_type,
            positional_embedding_size=config.visual.get("positional_embedding_size", 1024),
            mlp_ratio=config.visual.get("mlp_ratio", 4.9231),
            model_type=config.visual.get("model_type", "open_clip"),
            compute_dtype=config.compute_dtype,
            layernorm_compute_type=config.layernorm_compute_type,
        )

    def construct(
        self,
            input_ids,
            init_reset=True,
            batch_valid_length=None,
            batch_index=None,
            zactivate_len=None,
            windows=None,
            image=None,
    ):
        """construct"""
        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])

        # 1. wte
        bs, seq_len = self.shape(input_ids)
        hidden_states = self.wte(input_ids)

        # 2. drop
        hidden_states = self.drop(hidden_states)

        # image embedding
        if seq_len > 1 and image is not None and windows is not None:
            patch_list = []
            lora_idx = 0
            for image_patch in windows:
                patch = self.visual(image_patch, idx=lora_idx)
                patch_list.append(patch)
                lora_idx += 1
            global_feat = self.visual(image)

            local_feat = ops.cat(patch_list, axis=1)
            image_features = ops.cat([local_feat, global_feat], axis=1)

            if seq_len > 1 and image_features is not None:
                new_input_embeds = []
                num_patches = self.num_patches
                image_start_token_pos = self.image_start_token_pos
                for i in range(bs):
                    cur_input_embeds = hidden_states[i]
                    per_cur_image_features = image_features[i]
                    cur_input_embeds = ops.cat(
                        (
                            cur_input_embeds[: image_start_token_pos + 1],
                            per_cur_image_features,
                            cur_input_embeds[image_start_token_pos + num_patches + 1 :],
                        ),
                        axis=0,
                    )

                    new_input_embeds.append(cur_input_embeds)

                hidden_states = ops.stack(new_input_embeds, axis=0)

        # 3. rotary_emb
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
                    mask = self.casual_mask.increment_slice(
                        self.kvcache_preprocess.range,
                        self.kvcache_preprocess.max_cache_length // bs,
                        batch_valid_length,
                        zactivate_len,
                    )
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


@register_llm
class MonkeyQwenForCausalLM(QwenForCausalLM):
    def __init__(self, config):
        config = MonkeyConfig(**config)
        super().__init__(config)
        self.transformer = MonkeyModel(config)
        self.lm_head = Linear(
            config.hidden_size,
            config.vocab_size,
            has_bias=False,
            param_init_type=config.param_init_type,
            compute_dtype=config.compute_dtype
        )
        self.conversation = Conversation(generate_mode=True)
        self.image_processor = MonkeyImageProcessor()

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        image_path = kwargs.get("image_path")
        if image_path is None:
            windows, image = None, None
        else:
            windows, image = self.image_processor(image_path)
            windows = ms.Tensor(windows, ms.float16)
            image = ms.Tensor(image, ms.float16)
        return {
            "input_ids": ms.Tensor(input_ids, ms.int32),
            "windows": windows,
            "image": image,
        }

    def construct(
        self,
        input_ids,
        labels=None,
        input_position=None,
        position_ids=None,
        attention_mask=None,
        input_embeds=None,
        init_reset=True,
        batch_valid_length=None,
        batch_index=None,
        zactivate_len=None,
        windows=None,
        image=None,
    ):
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

        output = self.transformer(
            tokens,
            init_reset=init_reset,
            batch_valid_length=batch_valid_length,
            batch_index=batch_index,
            zactivate_len=zactivate_len,
            windows=windows,
            image=image,
        )
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

    def add_image_token_pad_in_query(self, query):
        query = self.image_prefix + " " + query + " "
        return query
