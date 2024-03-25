import argparse
import logging
import os

from PIL import Image

import mindspore as ms

from mindocr.data.transforms.llm_transform import image_processor, image_processor_high
from mindocr.nlp.llm.configs import LLMConfig
from mindocr.nlp.llm.qwen_tokenizer import QwenTokenizer
from mindocr.nlp.llm.vary_qwen_model import VaryQwenForCausalLM
from mindocr.utils.logger import set_logger


def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "1"):
        return True
    elif v.lower() in ("no", "false", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference Config Args")
    parser.add_argument("--image_dir", type=str, required=True, help="image path")
    parser.add_argument("--query", type=str, required=False, default="Provide the ocr results of this image.")
    parser.add_argument("--config_path", type=str, required=False, default="../../../configs/llm/vary/vary_toy.yaml")
    parser.add_argument("--chat_mode", type=str2bool, required=False, default=False)
    args = parser.parse_args()
    return args


class LLMGenerator(object):
    def __init__(self, args):
        config_path = args.config_path
        config = LLMConfig(config_path)
        ms.set_context(
            mode=ms.GRAPH_MODE,
            device_target="Ascend",
            enable_graph_kernel=False,
            graph_kernel_flags="--disable_expand_ops=Softmax,Dropout --enable_parallel_fusion=true "
            "--reduce_fuse_depth=8 --enable_auto_tensor_inplace=true",
            ascend_config={"precision_mode": "must_keep_origin_dtype"},
            max_call_depth=10000,
            max_device_memory="58GB",
            save_graphs=False,
            save_graphs_path="./graph",
            device_id=os.environ.get("DEVICE_ID", 0),
        )
        self.tokenizer = QwenTokenizer(**config.processor.tokenizer)
        self.model = VaryQwenForCausalLM.from_pretrained(config_path)

        self.image_dir = args.image_dir
        self.query = args.query
        self.seq_length = self.model.seq_length
        self.chat_mode = args.chat_mode

    def _call_one(self, query=None, image=None, image_high=None):
        response = self.model.chat(tokenizer=self.tokenizer, query=query, image=image, image_high=image_high)
        print(">" * 100)
        print(response)
        print("<" * 100)
        return response

    def __call__(self, query=None, image_dir=None):
        self.model.reset()
        is_first_iteration = True
        if query is None:
            query = self.query
        if image_dir is None:
            image_dir = self.image_dir
        image = load_image(image_dir)
        image_high = image_processor_high(image)
        image = image_processor(image)
        while True:
            try:
                if is_first_iteration:
                    self._call_one(query=query, image=image, image_high=image_high)
                    if not self.chat_mode:
                        break
                    is_first_iteration = False
                if self.chat_mode:
                    logging.info("You can input 'exit' to quit the conversation, or input your query:")
                    query = input()
                    if query == "exit":
                        break
                    self._call_one(query=query, image=None, image_high=None)
            except ValueError as e:
                if "check your inputs and set max_length larger than your inputs length." in e.args[0]:
                    logging.warning("The input is too long. The conversation is closed.")
                    break
                raise e


def main():
    set_logger()
    args = parse_args()
    llm_generator = LLMGenerator(args)
    llm_generator()


if __name__ == "__main__":
    main()
