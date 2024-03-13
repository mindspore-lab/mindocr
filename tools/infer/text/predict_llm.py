from PIL import Image

import mindspore as ms

from mindocr.models.llm.configs import LLMConfig
from mindocr.models.llm.qwen_tokenizer import QwenTokenizer
from mindocr.models.llm.vary_qwen_model import VaryQwenForCausalLM
from mindocr.data.transforms.llm_transform import image_processor_high, image_processor
from mindocr.utils.logger import set_logger
from config import create_parser

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'


def load_image(image_file):
    image = Image.open(image_file).convert('RGB')
    return image


def parse_args():
    parser = create_parser()
    parser.add_argument("--query", type=str, required=False, default="Provide the ocr results of this image.")
    parser.add_argument("--config_path", type=str, required=False, default="../../../configs/llm/vary/vary_toy.yaml")
    args = parser.parse_args()
    return args


def build_prompt(query):
    num_patch = 256
    prompt = '<|im_start|>system\nYou should follow the instructions carefully and explain your answers in detail.'
    prompt += '<|im_end|><|im_start|>user\n'
    prompt += DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * num_patch + DEFAULT_IM_END_TOKEN
    prompt += query + '<|im_end|><|im_start|>assistant\n'
    return prompt


class LLMGenerator(object):
    def __init__(self, args):
        config_path = args.config_path
        config = LLMConfig(config_path)
        ms.set_context(mode=0, device_target="Ascend", enable_graph_kernel=False,
                       graph_kernel_flags="--disable_expand_ops=Softmax,Dropout --enable_parallel_fusion=true "
                                          "--reduce_fuse_depth=8 --enable_auto_tensor_inplace=true",
                       ascend_config={"precision_mode": "must_keep_origin_dtype"}, max_call_depth=10000,
                       max_device_memory="58GB",
                       save_graphs=False,
                       save_graphs_path="./graph",
                       device_id=0)
        self.tokenizer = QwenTokenizer(**config.processor.tokenizer)
        self.model = VaryQwenForCausalLM.from_pretrained(config_path)

        self.image_dir = args.image_dir
        self.query = args.query
        self.seq_length = self.model.seq_length

    def __call__(self, query=None, image_dir=None):
        if query is None:
            query = self.query
        prompt = build_prompt(query)

        if image_dir is None:
            image_dir = self.image_dir
        image_dir = image_dir
        image = load_image(image_dir)
        image_high = image_processor_high(image)
        image = image_processor(image)

        inputs = self.tokenizer([prompt, ], max_length=self.seq_length)
        output = self.model.generate(input_ids=inputs["input_ids"], image=image, image_high=image_high)
        output_str = self.tokenizer.decode(output, skip_special_tokens=True)
        print('>' * 100)
        print('output:')
        print('-' * 100)
        for o in output_str:
            print(o)
        print('<' * 100)


def main():
    set_logger()
    args = parse_args()
    llm_generator = LLMGenerator(args)
    llm_generator()


if __name__ == '__main__':
    main()
