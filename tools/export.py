"""
Export ckpt files to mindir files for inference.

Args:
    model_name (str): Name of the model to be converted, or the path to the model YAML config file
    data_shape (int): The data shape [H, W] for exporting mindir files.
    local_ckpt_path (str): Path to a local checkpoint. If set, export mindir by loading local ckpt.
        Otherwise, export mindir by downloading online ckpt.
    save_dir (str): Directory to save the exported mindir file.

Example:
    >>> # Export mindir of model `dbnet_resnet50` by downloading online ckpt
    >>> python tools/export.py --model_name dbnet_resnet50 --data_shape 736 1280
    >>> # Export mindir of model `dbnet_resnet50` by loading local ckpt
    >>> python tools/export.py --model_name dbnet_resnet50 --data_shape 736 1280 --local_ckpt_path /path/to/local_ckpt
    >>> # Export mindir of model whose architecture is defined by crnn_resnet34.yaml with local checkpoint
    >>> python tools/export.py --model_name configs/rec/crnn/crnn_resnet34.yaml \
        --local_ckpt_path ~/.mindspore/models/crnn_resnet34-83f37f07.ckpt --data_shape 32 100

Notes:
    - Args `model_name` and `data_shape` are required to be specified when running export.py.
    - The `data_shape` is recommended to be the same as the rescaled data shape in evaluation to get the best inference
        performance.
    - The online ckpt files are downloaded from https://download.mindspore.cn/toolkits/mindocr/.
"""

import argparse
import os
import sys

import numpy as np
import yaml

import mindspore as ms

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from mindocr import build_model, list_models


def export(name_or_config, data_shape, local_ckpt_path, save_dir):
    ms.set_context(mode=ms.GRAPH_MODE)  # , device_target="Ascend")

    if name_or_config.endswith(".yml") or name_or_config.endswith(".yaml"):
        with open(name_or_config, "r") as f:
            cfg = yaml.safe_load(f)
            model_cfg = cfg["model"]
            amp_level = cfg["system"].get("amp_level_infer", "O0")
        name = os.path.basename(name_or_config).split(".")[0]
        assert (
            local_ckpt_path
        ), "Checkpoint path must be specified if using YAML config file to define model architecture. \
            Please set checkpoint path via `--local_ckpt_path`."
    else:
        model_cfg = name_or_config
        name = name_or_config
        amp_level = "O0"

    if local_ckpt_path:
        net = build_model(model_cfg, pretrained=False, ckpt_load_path=local_ckpt_path, amp_level=amp_level)
    else:
        net = build_model(model_cfg, pretrained=True, amp_level=amp_level)

    print(f"INFO: Set the AMP level of the model to be `{amp_level}`.")

    net.set_train(False)

    h, w = data_shape
    bs, c = 1, 3
    x = ms.Tensor(np.ones([bs, c, h, w]), dtype=ms.float32)

    output_path = os.path.join(save_dir, name) + ".mindir"
    ms.export(net, x, file_name=output_path, file_format="MINDIR")

    print(f"=> Finish exporting {name} to {os.path.realpath(output_path)}. The data shape [H, W] is {data_shape}")


def check_args(args):
    if args.local_ckpt_path and not os.path.isfile(args.local_ckpt_path):
        raise ValueError(f"Local ckpt file {args.local_ckpt_path} does not exist. Please check arg `local_ckpt_path`.")
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert model checkpoint to mindir format.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help=f"Name of the model to be converted or the path to model YAML config file. \
            Available choices for names: {list_models()}.",
    )
    parser.add_argument(
        "--data_shape",
        type=int,
        nargs=2,
        required=True,
        help="The data shape [H, W] for exporting mindir files. It is recommended to be the same as \
            the rescaled data shape in evaluation to get the best inference performance.",
    )
    parser.add_argument(
        "--local_ckpt_path",
        type=str,
        default="",
        help="Path to a local checkpoint. If set, export mindir by loading local ckpt. \
            Otherwise, export mindir by downloading online ckpt.",
    )
    parser.add_argument("--save_dir", type=str, default="", help="Directory to save the exported mindir file.")

    args = parser.parse_args()
    check_args(args)
    export(args.model_name, args.data_shape, args.local_ckpt_path, args.save_dir)
