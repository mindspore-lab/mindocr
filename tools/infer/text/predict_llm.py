import os
import sys

mindformers_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "../../../mindocr/models/third_party/mindformers"))
sys.path.insert(0, mindformers_path)
from PIL import Image

import mindspore as ms
from mindocr.models.third_party.mindformers.mindformers.tools import MindFormerConfig
from mindocr.models.third_party.mindformers.mindformers.core.context import build_context

from mindocr.models.third_party.mindformers.research.qwen.qwen_config import QwenConfig
from mindocr.models.third_party.mindformers.research.qwen import qwen_tokenizer
from mindocr.models.third_party.mindformers.research.qwen.qwen_tokenizer import QwenTokenizer

from mindocr.models.llm.vary_qwen_model import VaryQwenForCausalLM
from mindocr.data.transforms.llm_transform import image_processor_high, image_processor


DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


def load_image(image_file):
    image = Image.open(image_file).convert('RGB')
    return image


def main():
    qwen_tokenizer.SPECIAL_TOKENS += ('<ref>', '</ref>', '<box>', '</box>', '<quad>', '</quad>',
                                      '<img>', '</img>', '<imgpad>')
    config_file_path = "../../../mindocr/models/llm/run_vary_qwen_toy.yaml"

    config = MindFormerConfig(config_file_path)
    build_context(config)
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=4)

    tokenizer = QwenTokenizer(**config.processor.tokenizer)
    model_config = QwenConfig.from_pretrained(config_file_path)
    model = VaryQwenForCausalLM(model_config)

    img_path = './PMC5680412_00006.jpg'
    image = load_image(img_path)
    image_high = image_processor_high(image)
    image = image_processor(image)
    qs = 'Provide the ocr results of this image.'
    prompt = '<|im_start|>system\nYou should follow the instructions carefully and explain your answers in detail.'
    prompt += '<|im_end|><|im_start|>user\n'
    prompt += DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN
    prompt += qs + '<|im_end|><|im_start|>assistant\n'
    inputs = tokenizer([prompt, ], return_tensors=None, padding='max_length', max_length=model_config.seq_length)
    output = model.generate(input_ids=inputs["input_ids"], image=image, image_high=image_high)
    print(tokenizer.decode(output, skip_special_tokens=True))

    img_path = './PMC5680412_00006.jpg'
    image = load_image(img_path)
    image_high = image_processor_high(image)
    image = image_processor(image)
    qs = 'Describe this image in within 100 words.'
    prompt = '<|im_start|>system\nYou should follow the instructions carefully and explain your answers in detail.'
    prompt += '<|im_end|><|im_start|>user\n'
    prompt += DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN
    prompt += qs + '<|im_end|><|im_start|>assistant\n'
    inputs = tokenizer([prompt, ], return_tensors=None, padding='max_length', max_length=model_config.seq_length)
    output = model.generate(input_ids=inputs["input_ids"], image=image, image_high=image_high)
    print(tokenizer.decode(output, skip_special_tokens=True))


if __name__ == '__main__':
    main()
