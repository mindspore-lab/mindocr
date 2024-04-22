"""BaseModel"""
import os

from mindspore import nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindocr.nlp.generation import GeneratorMixin
from mindocr.nlp.llm.builder import build_llm_model
from mindocr.nlp.llm.configs import BaseConfig, LLMConfig


class BaseLLMModel(nn.Cell, GeneratorMixin):
    """
    The base model that contains the class method `from_pretrained` and `save_pretrained`, any new model that should
    inherit the class.

    Note:
        GeneratorMixin provides the method `generate` that enable the generation for nlp models.

    Args:
        config(BaseConfig): The model configuration that inherits the `BaseConfig`.
    """

    def __init__(self, config: BaseConfig, **kwargs):
        super(BaseLLMModel, self).__init__(**kwargs)
        self.config = config

    def load_checkpoint(self, config):
        """
        load checkpoint for models.

        Args:
            config (ModelConfig): a model config instance, which could have attribute
            "checkpoint_name_or_path (str)". set checkpoint_name_or_path to a supported
            model name or a path to checkpoint, to load model weights.
        """
        checkpoint_name_or_path = config.checkpoint_name_or_path
        if checkpoint_name_or_path:
            if not isinstance(checkpoint_name_or_path, str):
                raise TypeError(f"checkpoint_name_or_path should be a str, but got {type(checkpoint_name_or_path)}")

            if os.path.exists(checkpoint_name_or_path):
                param = load_checkpoint(checkpoint_name_or_path)
            else:
                raise ValueError(
                    f"{checkpoint_name_or_path} is not a supported default model"
                    f" or a valid path to checkpoint,"
                    f" please select from {self._support_list}."
                )

            load_param_into_net(self, param)

    @classmethod
    def _get_config_args(cls, pretrained_model_name_or_dir, **kwargs):
        """build config args."""
        is_dir = os.path.isdir(pretrained_model_name_or_dir)

        if is_dir:
            yaml_list = [file for file in os.listdir(pretrained_model_name_or_dir) if file.endswith(".yaml")]
            yaml_list.sort()
            config_args = None
            for yaml_file in yaml_list:
                if config_args is None:
                    config_args = LLMConfig(yaml_file)
                else:
                    sub_config_args = LLMConfig(yaml_file)
                    config_args.model.update(**sub_config_args)
            config_args.model.update(**kwargs)
        else:
            yaml_file = pretrained_model_name_or_dir
            config_args = LLMConfig(yaml_file)
            config_args.model.update(**kwargs)
        return config_args

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_dir: str, **kwargs):
        if not isinstance(pretrained_model_name_or_dir, str):
            raise TypeError(
                f"pretrained_model_name_or_dir should be a str, but got {type(pretrained_model_name_or_dir)}"
            )
        config_args = cls._get_config_args(pretrained_model_name_or_dir, **kwargs)
        model = build_llm_model(config_args.model)
        return model
