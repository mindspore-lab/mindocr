import copy
import os

import yaml

import mindspore.common.dtype as mstype


def convert_mstype(ms_type: str = "float16"):
    """Convert the string type to MindSpore type."""
    if isinstance(ms_type, mstype.Float):
        return ms_type
    if ms_type == "float16":
        return mstype.float16
    if ms_type == "bfloat16":
        return mstype.bfloat16
    if ms_type == "float32":
        return mstype.float32
    raise KeyError(f"Supported data type keywords include: [float16, float32, bfloat16], but get {ms_type}")


class LLMConfig(dict):
    def __init__(self, *args, **kwargs):
        super(LLMConfig, self).__init__()
        cfg_dict = {}

        # load from file
        for arg in args:
            if isinstance(arg, str):
                if arg.endswith("yaml") or arg.endswith("yml"):
                    raw_dict = LLMConfig._file2dict(arg)
                    cfg_dict.update(raw_dict)

        # load dictionary configs
        if kwargs is not None:
            cfg_dict.update(kwargs)

        LLMConfig._dict2config(self, cfg_dict)

    def __getattr__(self, key):
        """Get a object attr by its `key`

        Args:
            key (str) : the name of object attr.

        Returns:
            attr of object that name is `key`
        """
        if key not in self:
            return None

        return self[key]

    def __setattr__(self, key, value):
        """Set a object value `key` with `value`

        Args:
            key (str) : The name of object attr.
            value : the `value` need to set to the target object attr.
        """
        self[key] = value

    def __delattr__(self, key):
        """Delete a object attr by its `key`.

        Args:
            key (str) : The name of object attr.
        """
        del self[key]

    def __deepcopy__(self):
        """Deep copy operation on arbitrary LLMConfig objects.

        Returns:
            LLMConfig : The deep copy of the given LLMConfig object.
        """
        config = LLMConfig()
        for key in self.keys():
            config.__setattr__(copy.deepcopy(key), copy.deepcopy(self.__getattr__(key)))
        return config

    @staticmethod
    def _file2dict(filename=None):
        """Convert config file to dictionary.

        Args:
            filename (str) : config file.
        """
        if filename is None:
            raise NameError("This {} cannot be empty.".format(filename))

        filepath = os.path.realpath(filename)
        with open(filepath, encoding="utf-8") as fp:
            cfg_dict = yaml.load(fp, yaml.Loader)

        return cfg_dict

    @staticmethod
    def _dict2config(config, dic):
        """Convert dictionary to config.

        Args:
            config : Config object
            dic (dict) : dictionary
        Returns:

        Exceptions:

        """
        if isinstance(dic, dict):
            for key, value in dic.items():
                if isinstance(value, dict):
                    sub_config = LLMConfig()
                    dict.__setitem__(config, key, sub_config)
                    LLMConfig._dict2config(sub_config, value)
                else:
                    config[key] = dic[key]


class BaseConfig(dict):
    def __init__(self, **kwargs):
        super(BaseConfig, self).__init__()
        self.update(kwargs)

    def __getattr__(self, key):
        if key not in self:
            return None
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    @classmethod
    def from_pretrained(cls, yaml_name_or_path, **kwargs):
        """
        From pretrain method, which instantiates a config by yaml name or path.

        Args:
            yaml_name_or_path (str): A supported model path to model config (.yaml).

        Returns:
            A model config, which inherited from BaseConfig.
        """
        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)
        if pretrained_model_name_or_path is not None:
            yaml_name_or_path = pretrained_model_name_or_path

        if not isinstance(yaml_name_or_path, str):
            raise TypeError(f"yaml_name_or_path should be a str, but got {type(yaml_name_or_path)}.")

        if os.path.exists(yaml_name_or_path):
            if not yaml_name_or_path.endswith(".yaml"):
                raise ValueError(f"{yaml_name_or_path} should be a .yaml file for model config.")

            config_args = LLMConfig(yaml_name_or_path)
        else:
            raise ValueError(f"{yaml_name_or_path} is not a supported model type or a valid path to model config.")
        config_args.model.update(**kwargs)
        config = config_args.model
        return config


class QwenConfig(BaseConfig):
    def __init__(
        self,
        batch_size: int = 1,
        seq_length: int = 2048,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        n_kv_heads: int = None,
        max_position_embedding: int = None,
        intermediate_size: int = None,
        vocab_size: int = 32000,  # defined later by tokenizer
        multiple_of: int = 256,  # make SwiGLU hidden layer size multiple of large power of 2
        ffn_dim_multiplier: int = None,
        rms_norm_eps: float = 1e-5,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        ignore_token_id: int = -100,
        theta: float = 10000.0,
        compute_dtype: str = "float16",
        layernorm_compute_type: str = "float32",
        softmax_compute_type: str = "float32",
        rotary_dtype: str = "float32",
        param_init_type: str = "float16",
        ln_param_init_type: str = "float32",
        qkv_has_bias: bool = False,
        qkv_concat: bool = False,
        use_past: bool = False,
        pretrain_seqlen=None,
        extend_method: str = "None",
        scaling_factor: float = 1.0,
        is_dynamic: bool = False,
        use_kvcache_op: bool = False,
        is_flexible_shape: bool = False,
        use_rope_slice: bool = False,
        use_flash_attention: bool = False,
        use_paged_attention: bool = False,
        fine_grain_interleave: int = 1,
        offset: int = 0,
        checkpoint_name_or_path: str = "",
        repetition_penalty: float = 1.0,
        max_decode_length: int = 1024,
        block_size: int = 16,
        num_blocks: int = 512,
        top_k: int = 5,
        top_p: float = 1.0,
        do_sample: bool = True,
        **kwargs,
    ):
        super(QwenConfig, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_position_embedding = max_position_embedding if max_position_embedding else seq_length
        self.intermediate_size = intermediate_size
        self.multiple_of = multiple_of
        self.n_kv_heads = n_kv_heads
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.rms_norm_eps = rms_norm_eps
        self.qkv_concat = qkv_concat
        self.param_init_type = convert_mstype(param_init_type)
        self.qkv_has_bias = qkv_has_bias
        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.rotary_dtype = convert_mstype(rotary_dtype)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.ln_param_init_type = convert_mstype(ln_param_init_type)
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.ignore_token_id = ignore_token_id
        self.use_past = use_past
        self.pretrain_seqlen = pretrain_seqlen
        self.extend_method = extend_method
        self.scaling_factor = scaling_factor
        self.is_dynamic = is_dynamic
        self.use_kvcache_op = use_kvcache_op
        self.is_flexible_shape = is_flexible_shape
        self.use_rope_slice = use_rope_slice
        self.use_flash_attention = use_flash_attention
        self.fine_grain_interleave = fine_grain_interleave
        self.offset = offset
        self.repetition_penalty = repetition_penalty
        self.max_decode_length = max_decode_length
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.theta = theta
        self.use_paged_attention = use_paged_attention
        self.block_size = block_size
        self.num_blocks = num_blocks


class VaryConfig(QwenConfig):
    def __init__(self, **kwargs):
        super(VaryConfig, self).__init__(**kwargs)


class SAMConfig(BaseConfig):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        out_chans: int = 256,
        qkv_bias: bool = True,
        layer_norm_eps: float = 1.0e-6,
        use_abs_pos: bool = True,
        use_rel_pos: bool = True,
        rel_pos_zero_init: bool = True,
        window_size: int = 14,
        global_attn_indexes: tuple = (2, 5, 8, 11),
        checkpoint_name_or_path: str = "",
        compute_dtype: str = "float16",
        layernorm_compute_type: str = "float32",
        softmax_compute_type: str = "float16",
        param_init_type: str = "float16",
        ln_param_init_type: str = "float32",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.out_chans = out_chans
        self.qkv_bias = qkv_bias
        self.layer_norm_eps = layer_norm_eps
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        self.rel_pos_zero_init = rel_pos_zero_init
        self.window_size = window_size
        self.global_attn_indexes = global_attn_indexes

        self.param_init_type = convert_mstype(param_init_type)
        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.ln_param_init_type = convert_mstype(ln_param_init_type)

        self.checkpoint_name_or_path = checkpoint_name_or_path
