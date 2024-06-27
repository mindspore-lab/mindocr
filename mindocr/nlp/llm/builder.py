from ._registry import is_llm, is_llm_class, list_llms, llm_class_entrypoint, llm_entrypoint
from .configs import LLMConfig

__all__ = ["build_llm_model"]


def build_llm_model(config):
    """

    Example:
        >>> from mindocr.nlp.llm import build_llm_model
        >>> llm_model = build_llm_model(dict(name='VaryQwenForCausalLM'))
        >>> print(llm_model)
    """
    if isinstance(config, dict):
        config = LLMConfig(**config)
    elif isinstance(config, str):
        config = LLMConfig(config)
    else:
        raise TypeError(f"config must be str or dict, but got {type(config)}")
    if "model" in config:
        config = LLMConfig(**config["model"], **config["processor"])
    if "name" not in config:
        raise ValueError("`name` must in `config`.")
    name = config["name"]
    if is_llm(name):
        create_fn = llm_entrypoint(name)
        llm = create_fn(config)
    elif is_llm_class(name):
        llm_class = llm_class_entrypoint(name)
        llm = llm_class(config)
    else:
        raise ValueError(f"Invalid llm name: {name}, supported llms are: {list_llms()}")

    if "checkpoint_name_or_path" in config:
        llm.load_checkpoint(config)

    return llm
