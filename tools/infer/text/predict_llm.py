import os
import sys

mindformers_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "../../../mindocr/models/third_party/mindformers"))
sys.path.insert(0, mindformers_path)

import mindspore as ms
from mindformers import MindFormerConfig
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config

from mindocr.models.llm.vary_qwen_model import VaryQwenForCausalLM

from mindocr.models.third_party.mindformers.research.qwen.qwen_config import QwenConfig
from mindocr.models.third_party.mindformers.research.qwen import qwen_tokenizer
from mindocr.models.third_party.mindformers.research.qwen.qwen_tokenizer import QwenTokenizer

qwen_tokenizer.SPECIAL_TOKENS += ('<ref>', '</ref>', '<box>', '</box>', '<quad>', '</quad>',
                                  '<img>', '</img>', '<imgpad>')
config_file_path = "../../../mindocr/models/llm/run_vary_qwen_toy.yaml"
config = MindFormerConfig(config_file_path)
build_context(config)
build_parallel_config(config)
# ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=1)
tokenizer = QwenTokenizer(**config.processor.tokenizer)

model_config = QwenConfig.from_pretrained(config_file_path)
# model = QwenForCausalLM(model_config)
model = VaryQwenForCausalLM(model_config)

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'
DEFAULT_PAD_TOKEN = "<|endoftext|>"
prompt = '<|im_start|>system\nYou should follow the instructions carefully and explain your answers in detail.'
prompt += '<|im_end|><|im_start|>user\n'
prompt += '<img>' + '<imgpad>' * 256 + '</img>'
prompt += 'Provide the ocr results of this image.<|im_end|><|im_start|>assistant\n'
inputs = tokenizer([prompt, ], return_tensors=None, padding='max_length', max_length=model_config.seq_length)
output = model.generate(input_ids=inputs["input_ids"])
print(tokenizer.decode(output, skip_special_tokens=True))
