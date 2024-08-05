import argparse
import logging
import os

import mindspore as ms

from mindocr.nlp.llm import build_llm_model
from mindocr.utils.logger import set_logger


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
    parser.add_argument("--mode", type=int, default=0, help="0 for graph mode, 1 for pynative mode")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument("--image_path", type=str, required=False, help="image path")
    parser.add_argument("--query", type=str, required=False, default="Provide the ocr results of this image.")
    parser.add_argument("--config_path", type=str, required=False, default="../../../configs/llm/vary/vary_toy.yaml")
    parser.add_argument("--chat_mode", type=str2bool, required=False, default=False)
    parser.add_argument("--precision_mode", type=str, required=False, default="allow_fp32_to_fp16")
    args = parser.parse_args()
    return args


class LLMGenerator(object):
    def __init__(self, args):
        config_path = args.config_path
        ms.set_context(
            mode=args.mode,
            device_target="Ascend",
            enable_graph_kernel=False,
            graph_kernel_flags="--disable_expand_ops=Softmax,Dropout --enable_parallel_fusion=true "
            "--reduce_fuse_depth=8 --enable_auto_tensor_inplace=true",
            ascend_config={"precision_mode": args.precision_mode},
            max_call_depth=10000,
            max_device_memory="58GB",
            save_graphs=False,
            save_graphs_path="./graph",
            device_id=os.environ.get("DEVICE_ID", 0),
        )
        self.model = build_llm_model(config_path)

        self.image_path = args.image_path
        self.query = args.query
        self.seq_length = self.model.seq_length
        self.chat_mode = args.chat_mode

    def _call_one(self, query=None, image_path=None):
        response = self.model.chat(query=query, image_path=image_path)
        print(">" * 100)
        print(response)
        print("<" * 100)
        return response

    def __call__(self, query=None, image_path=None):
        self.model.reset()
        is_first_iteration = True
        if query is None:
            query = self.query
        if image_path is None:
            image_path = self.image_path
        while True:
            try:
                if is_first_iteration:
                    self._call_one(query=query, image_path=image_path)
                    if not self.chat_mode:
                        break
                    is_first_iteration = False
                if self.chat_mode:
                    logging.info("You can input 'exit' to quit the conversation, or input your query:")
                    query = input()
                    if query == "exit":
                        break
                    self._call_one(query=query, image_path=image_path)
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
