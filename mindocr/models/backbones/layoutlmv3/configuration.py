from dataclasses import dataclass


@dataclass
class LayoutLMv3PretrainedConfig:
    def __init__(self, use_float16=False, **kwargs):
        pretrained_config = {
            "use_float16": use_float16,
            "fast_qkv": False,
            "vocab_size": kwargs.get("vocab_size", 250002),
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 514,
            "type_vocab_size": 1,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-5,
            "pad_token_id": 1,
            "bos_token_id": 0,
            "eos_token_id": 2,
            "max_2d_position_embeddings": 1024,
            "coordinate_size": 128,
            "shape_size": 128,
            "has_relative_attention_bias": True,
            "rel_pos_bins": 32,
            "max_rel_pos": 128,
            "rel_2d_pos_bins": 64,
            "max_rel_2d_pos": 256,
            "has_spatial_attention_bias": True,
            "text_embed": True,
            "visual_embed": True,
            "input_size": 224,
            "num_channels": 3,
            "patch_size": 16,
            "classifier_dropout": None,
            "num_labels": None,
        }

        for key, value in pretrained_config.items():
            setattr(self, key, value)
