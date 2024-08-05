"""BaseModel"""
import os

from mindspore import nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindocr.nlp.generation import GeneratorMixin
from mindocr.nlp.llm.builder import build_llm_model
from mindocr.nlp.llm.configs import BaseConfig, LLMConfig
from mindocr.utils.conversation import Conversation


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
        self.conversation = Conversation()
        self.image_path = None
        self.IMAGE_START_TAG = "<img>"
        self.IMAGE_END_TAG = "</img>"
        self.IMAGE_PAD_TAG = "<imgpad>"
        self.num_patches = config.num_patches
        self.image_prefix = f"{self.IMAGE_START_TAG}{self.IMAGE_PAD_TAG * self.num_patches}{self.IMAGE_END_TAG}"

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
                raise ValueError(f"{checkpoint_name_or_path} is not a valid path to checkpoint.")

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

    def reset(self):
        self.conversation.messages = list()

    def add_image_token_pad_in_query(self, query: str):
        query = self.image_prefix + query
        return query

    def chat(
        self,
        query: str,
        image_path: str = None,
    ) -> str:
        """
        If `image_path` is provided, the conversation will be reset.
        example:
            inputs:
                query: Provide the ocr results of this image.
                image_path: xx/xxx.png.
            outputs:
                response: the modalities of irradiation could be modified...
                history: [
                    ("user", "Provide the ocr results of this image."),
                    ("assistant", "the modalities of irradiation could be modified..."),
                ]

        """
        if image_path is not None:
            query = self.add_image_token_pad_in_query(query=query)
            self.image_path = image_path
            self.reset()

        self.conversation.add_message(role="user", message=query)
        prompt = self.conversation.get_prompt()

        inputs = self.tokenizer([prompt], max_length=self.seq_length)
        input_ids = inputs["input_ids"]
        outputs = self.generate(input_ids=input_ids, image_path=self.image_path)
        outputs = self.tokenizer.decode(outputs, skip_special_tokens=False)
        response = outputs[0][len(prompt) :]

        for special_token in self.tokenizer.special_tokens:
            response = response.replace(special_token, "")
        self.conversation.add_message(role="assistant", message=response)

        return response
