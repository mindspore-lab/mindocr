"""generation config."""
import copy
import logging
from typing import Any, Dict

__all__ = ["GenerationConfig"]
_logger = logging.getLogger(__name__)


class GenerationConfig:
    """Class that holds a configuration for a generation task.
    Args:
        > Parameters that control the length of the output

        max_length (`int`, *optional*, defaults to 20):
            The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
            `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
        max_new_tokens (`int`, *optional*):
            The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.

        > Parameters that control the generation strategy used

        do_sample (`bool`, *optional*, defaults to `False`):
            Whether to use sampling ; use greedy decoding otherwise.
        use_past (`bool`, *optional*, defaults to `False`):
            Whether the model should use the past last key/values attentions
            (if applicable to the model) to speed up decoding.
        num_beams(`int`, *optional*, defaults to 1):
            Number of beams for beam search. 1 means no beam search. If larger than 1, use beam search strategy.

        > Parameters for manipulation of the model output logits

        temperature (`float`, *optional*, defaults to 1.0):
            The value used to modulate the next token probabilities.
        top_k (`int`, *optional*, defaults to 50):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (`float`, *optional*, defaults to 1.0):
            If set to float < 1, only the smallest set of most probable tokens with probabilities
            that add up to `top_p` or higher are kept for generation.
        repetition_penalty (`float`, *optional*, defaults to 1.0):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
        encoder_repetition_penalty (`float`, *optional*, defaults to 1.0):
            The parameter for encoder_repetition_penalty. An exponential penalty on sequences
            that are not in the original input. 1.0 means no penalty.
        renormalize_logits (`bool`, *optional*, defaults to `False`):
            Whether to renormalize the logits after applying all the logits processors or wrappers (including the custom
            ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the score logits
            are normalized but some logit processors or wrappers break the normalization.

        > Special tokens that can be used at generation time

        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        bos_token_id (`int`, *optional*):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to
            set multiple *end-of-sequence* tokens.

        > Wild card

        generation_kwargs:
            Additional generation kwargs will be forwarded to the `generate` function of the model.
            Kwargs that are not present in `generate`'s signature will be used in the
            model forward pass.
    """

    def __init__(self, **kwargs):
        # max generate length
        self.max_length = kwargs.pop("max_decode_length", 20)
        self.max_length = kwargs.pop("max_length", self.max_length)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)

        # number of beams
        self.num_beams = kwargs.pop("num_beams", 1)
        # do sample or not
        self.do_sample = kwargs.pop("do_sample", False)
        # incremental infer
        self.use_past = kwargs.pop("use_past", False)
        # logits processors
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.encoder_repetition_penalty = kwargs.pop("encoder_repetition_penalty", 1.0)
        self.renormalize_logits = kwargs.pop("renormalize_logits", False)

        # Special tokens that can be used at generation time
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config
            # if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    _logger.error("Can't set %s with value %s for %s", key, value, self)
                    raise err

    def __str__(self) -> str:
        return str(self.__dict__)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        """
        Instantiates a [`GenerationConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            kwargs:
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`GenerationConfig`]: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        config = cls(**{**config_dict, **kwargs})
        unused_kwargs = config.update(**kwargs)
        _logger.debug("Generate config %s", config)
        if return_unused_kwargs:
            return config, unused_kwargs
        return config

    @classmethod
    def from_model_config(cls, model_config) -> "GenerationConfig":
        config_dict = model_config
        config_dict.pop("_from_model_config", None)
        config = cls.from_dict(config_dict, return_unused_kwargs=False, _from_model_config=True)

        return config

    def update(self, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs`
        if they match existing attributes, returning all the unused kwargs.

        Args:
            kwargs (`Dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `Dict[str, Any]`: Dictionary containing all the key-value pairs
            that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs

    def to_dict(self) -> Dict[str, Any]:
        """to dict convert function."""
        output = copy.deepcopy(self.__dict__)
        return output
