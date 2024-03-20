import mindspore as ms
from mindspore import ops

from mindocr.nlp.llm import register_llm
from mindocr.nlp.llm.configs import SAMConfig, VaryConfig
from mindocr.nlp.llm.qwen_model import QwenForCausalLM, QwenModel
from mindocr.nlp.llm.vary_clip_model import build_model
from mindocr.nlp.llm.vary_sam_model import SAMEncoder
from mindocr.nlp.utils.layers import Linear
from mindocr.utils.conversation import Conversation


class VaryQwenModel(QwenModel):
    def __init__(self, config):
        super(VaryQwenModel, self).__init__(config)
        config = SAMConfig(ln_param_init_type=config.ln_param_init_type)
        self.vision_tower_high = SAMEncoder(config)
        self.vision_tower_high.to_float(ms.float16)

        self.vision_tower = build_model(
            param_init_type=config.param_init_type, ln_param_init_type=config.ln_param_init_type
        )
        self.vision_tower.to_float(ms.float16)

        self.mm_projector = Linear(1024, 1024, param_init_type=config.param_init_type)
        self.mm_projector_vary = Linear(1024, 1024, param_init_type=config.param_init_type)

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
        bs, seq_len = self.shape(input_ids)
        inputs_embeds = self.wte(input_ids)

        if seq_len > 1 and image is not None and image_high is not None:
            sam_out = self.vision_tower_high(image_high)
            sam_out = self.mm_projector_vary(sam_out)

            clip_out = self.vision_tower(image)
            clip_out = self.mm_projector(clip_out)

            image_features = ops.concat((clip_out, sam_out), -1)

            new_input_embeds = []
            num_patches = self.num_patches
            image_start_token_pos = self.image_start_token_pos
            for i in range(bs):
                cur_input_embeds = inputs_embeds[i]
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
        else:
            hidden_states = inputs_embeds

        # 2. drop
        hidden_states = self.drop(hidden_states)

        # 2. rotary_emb
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
class VaryQwenForCausalLM(QwenForCausalLM):
    def __init__(self, config):
        config = VaryConfig(**config)
        super(VaryQwenForCausalLM, self).__init__(config)
        self.transformer = VaryQwenModel(config=config)
        self.conversation = None

        self.image_past = None
        self.image_high_past = None

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        image = kwargs.get("image")
        image_high = kwargs.get("image_high")
        return {
            "input_ids": ms.Tensor(input_ids, ms.int32),
            "image": ms.Tensor(image, ms.float16) if image is not None else None,
            "image_high": ms.Tensor(image_high, ms.float16) if image_high is not None else None,
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
        image=None,
        image_high=None,
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
            image=image,
            image_high=image_high,
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

    def chat(
        self,
        tokenizer,
        query: str,
        image=None,
        image_high=None,
    ) -> str:
        """
        example:
            inputs:
                query: Provide the ocr results of this image.
                image: np.array.
                image_high: np.array.
            outputs:
                response: the modalities of irradiation could be modified...
                history: [
                    ("user", "Provide the ocr results of this image."),
                    ("assistant", "the modalities of irradiation could be modified..."),
                ]

        """
        if self.conversation is None:
            self.conversation = Conversation()

        if image is not None and image_high is not None:
            num_patch = 256
            im_start_token = "<img>"
            im_end_token = "</img>"
            im_patch_token = "<imgpad>"
            query = im_start_token + im_patch_token * num_patch + im_end_token + query
            self.image_past = image
            self.image_high_past = image_high

        self.conversation.add_message(role="user", message=query)
        prompt = self.conversation.get_prompt()

        inputs = tokenizer([prompt], max_length=self.seq_length)
        input_ids = inputs["input_ids"]
        outputs = self.generate(input_ids=input_ids, image=self.image_past, image_high=self.image_high_past)
        outputs = tokenizer.decode(outputs, skip_special_tokens=False)
        response = outputs[0][len(prompt) :]

        for special_token in tokenizer.special_tokens:
            response = response.replace(special_token, "")
        self.conversation.add_message(role="assistant", message=response)

        return response

    def reset(self):
        if self.conversation is not None:
            self.conversation.messages = list()
