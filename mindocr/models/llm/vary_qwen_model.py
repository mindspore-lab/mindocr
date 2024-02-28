import numpy as np

import mindspore as ms
from mindspore import ops
from mindspore import nn

from mindocr.models.third_party.mindformers.research.qwen.qwen_model import QwenForCausalLM, QwenModel
from mindocr.models.third_party.mindformers.research.qwen.qwen_config import QwenConfig


class VaryConfig(QwenConfig):
    model_type = "vary"


class VaryQwenModel(QwenModel):
    def __init__(self, config: QwenConfig):
        super(VaryQwenModel, self).__init__(config)
        # self.vision_tower = CLIPVisionModel.from_pretrained(
        #     '/ms_test3/psw/workspace/Vary/Vary-master/vit-large-patch14')
        #
        # self.vision_tower_high = build_sam_vit_b()  # build_sam_vit_b(checkpoint = 'xxxx') for train

        self.mm_projector = nn.Dense(1024, 1024)
        self.mm_projector_vary = nn.Dense(1024, 1024)

        self.vision_select_layer = getattr(self.config, "vision_select_layer", -1)

        self.image_start_token_pos = 22
        self.num_patches = 256

    # def initialize_vision_modules(
    #         self,
    #         vision_tower,
    #         pretrained_stage1_model=None,
    #         freeze_vision_tower=False,
    #         use_im_start_end=False,
    #         vision_select_layer=-1,
    #         dtype=ms.float16,
    #         device="cuda"
    # ):
    #
    #     # 224*224
    #     image_processor = CLIPImageProcessor.from_pretrained(
    #         '/ms_test3/psw/workspace/Vary/Vary-master/vit-large-patch14')
    #     # 1024*1024
    #     image_processor_high = train_transform
    #
    #     self.vision_tower = self.vision_tower.to(dtype=dtype, device=device)
    #
    #     self.vision_tower_high = self.vision_tower_high.to(dtype=dtype, device=device)
    #
    #     self.mm_projector = self.mm_projector.to(dtype=dtype, device=device)
    #     self.mm_projector_vary = self.mm_projector_vary.to(dtype=dtype, device=device)
    #
    #     image_token_len = 256
    #
    #     self.config.vision_tower = vision_tower
    #     self.config.image_token_len = image_token_len
    #
    #     self.config.use_im_start_end = True
    #
    #     self.config.vision_select_layer = vision_select_layer
    #     self.config.freeze_vision_tower = freeze_vision_tower
    #
    #     return dict(
    #         image_processor=image_processor,
    #         image_processor_high=image_processor_high,
    #         image_token_len=image_token_len,
    #
    #     )

    def construct(
            self,
            input_ids,
            init_reset=True,
            batch_valid_length=None,
            batch_index=None,
            zactivate_len=None,
            images=None,
    ):
        # 1. wte
        input_shape = input_ids.shape
        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = self.wte(input_ids)

        # vision_tower = self.vision_tower
        # vision_tower_high = self.vision_tower_high
        # vision_select_layer = self.vision_select_layer

        # image_features_1 = []
        # image_features_2 = []
        # for image in images:
        #     image_forward_out = vision_tower(image[0], output_hidden_states=True)
        #     select_hidden_state = image_forward_out.hidden_states[vision_select_layer]
        #     image_feature = select_hidden_state[:, 1:]  # 256*1024
        #     cnn_feature = vision_tower_high(image[1])
        #     cnn_feature = cnn_feature.flatten(2).permute(0, 2, 1)  # 256*1024
        #
        #     image_features_1.append(image_feature)
        #     image_features_2.append(cnn_feature)
        #
        # image_features_1 = [self.mm_projector(image_feature) for image_feature in image_features_1]
        # image_features_2 = [self.mm_projector_vary(image_feature) for image_feature in image_features_2]
        # image_features = [ops.cat((image_feature[0], image_feature[1]), axis=-1) for image_feature in
        #                   zip(image_features_1, image_features_2)]
        image_features = ms.Tensor(np.load('./image_features.npy').astype(np.float16))

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
