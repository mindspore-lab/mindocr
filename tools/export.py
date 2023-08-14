"""
Export ckpt files to mindir files for inference.

Args:
    model_name_or_config (str): Name of the model to be converted, or the path to the model YAML config file
    data_shape (int): The data shape [H, W] for exporting mindir files.
    local_ckpt_path (str): Path to a local checkpoint. If set, export mindir by loading local ckpt.
        Otherwise, export mindir by downloading online ckpt.
    save_dir (str): Directory to save the exported mindir file.

Example:
    >>> # Export mindir of model `dbnet_resnet50` by downloading online ckpt
    >>> python tools/export.py --model_name_or_config dbnet_resnet50 --data_shape 736 1280
    >>> # Export mindir of model `dbnet_resnet50` by loading local ckpt
    >>> python tools/export.py --model_name_or_config dbnet_resnet50 --data_shape 736 1280 \
        --local_ckpt_path /path/to/dbnet.ckpt
    >>> # Export mindir of model whose architecture is defined by crnn_resnet34.yaml with local checkpoint
    >>> python tools/export.py --model_name_or_config configs/rec/crnn/crnn_resnet34.yaml \
        --local_ckpt_path ~/.mindspore/models/crnn_resnet34-83f37f07.ckpt --data_shape 32 100
    >>> # Export mindir with dynamic input data shape.
          Dynamic input data shape of detection model: (-1,3,-1,-1)
          Dynamic input data shape of recognition and classification model: (-1,3,32,-1)
    >>> python tools/export.py --model_name_or_config configs/rec/crnn/crnn_resnet34.yaml --is_dynamic_shape True \
        --model_type rec --local_ckpt_path path/to/crnn.ckpt

Notes:
    - Arg `model_name_or_config` is required to be specified when running export.py.
    - Arg `data_shape` is recommended to be the same as the rescaled data shape in evaluation to get the best inference
        performance.
    - When arg `is_dynamic_shape` is False (default value is False), arg `data_shape` is required to be specified.
    - When arg `is_dynamic_shape` is True, arg `model_type` is required to be specified.
    - The online ckpt files are downloaded from https://download.mindspore.cn/toolkits/mindocr/.
"""
import argparse
import logging
import os
import sys

import numpy as np
import yaml

import mindspore as ms

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from mindocr import build_model, list_models
from mindocr.utils.logger import set_logger

logger = logging.getLogger("mindocr.export")


def export(model_name_or_config, data_shape, local_ckpt_path, save_dir, is_dynamic_shape, model_type, **kwargs):
    ms.set_context(mode=ms.GRAPH_MODE)  # , device_target="Ascend")
    set_logger(name="mindocr")

    if model_name_or_config.endswith(".yml") or model_name_or_config.endswith(".yaml"):
        with open(model_name_or_config, "r") as f:
            cfg = yaml.safe_load(f)
            model_cfg = cfg["model"]
            amp_level = cfg["system"].get("amp_level_infer", "O0")
        name = os.path.basename(model_name_or_config).rsplit(".", 1)[0]
    else:
        model_cfg = model_name_or_config
        name = model_name_or_config
        amp_level = "O0"

    if local_ckpt_path:
        net = build_model(model_cfg, pretrained=False, ckpt_load_path=local_ckpt_path, amp_level=amp_level)
    else:
        net = build_model(model_cfg, pretrained=True, amp_level=amp_level)

    logger.info(f"Set the AMP level of the model to be `{amp_level}`.")

    net.set_train(False)

    if is_dynamic_shape:
        if model_type == "det":
            x = ms.Tensor(shape=[None, 3, None, None], dtype=ms.float32)
        else:
            x = ms.Tensor(shape=[None, 3, 32, None], dtype=ms.float32)
    else:
        h, w = data_shape
        bs, c = 1, 3
        x = ms.Tensor(np.ones([bs, c, h, w]), dtype=ms.float32)

    output_path = os.path.join(save_dir, name) + ".mindir"
    ms.export(net, x, file_name=output_path, file_format="MINDIR")

    logger.info(
        f"=> Finish exporting mindir file of {name} to {os.path.realpath(output_path)}."
        f"The data shape (N, C, H, W) is {x.shape}."
    )


def check_args(args):
    if args.model_name_or_config.endswith(".yml") or args.model_name_or_config.endswith(".yaml"):
        assert os.path.isfile(
            args.model_name_or_config
        ), f"YAML config file '{args.model_name_or_config}' does not exist. Please check arg `model_name_or_config`."
        assert (
            args.local_ckpt_path is not None
        ), "Local checkpoint path must be specified if using YAML config file to define model architecture. \
        Please set arg `local_ckpt_path`."
    if args.local_ckpt_path and not os.path.isfile(args.local_ckpt_path):
        raise ValueError(
            f"Local ckpt file '{args.local_ckpt_path}' does not exist. Please check arg `local_ckpt_path`."
        )
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    if args.is_dynamic_shape:
        assert (
            args.model_type is not None
        ), "You are exporting mindir with dynamic data shape. Please set arg `model_type` as det, rec or cls."
    else:
        assert (
            args.data_shape is not None
        ), "You are exporting mindir with static data shape. Please set arg `data_shape`."


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "1"):
        return True
    elif v.lower() in ("no", "false", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert model checkpoint to mindir format.")
    parser.add_argument(
        "--model_name_or_config",
        type=str,
        required=True,
        help=f"Name of the model to be converted or the path to model YAML config file. \
            Available choices for names: {list_models()}.",
    )
    parser.add_argument(
        "--is_dynamic_shape",
        type=str2bool,
        default=False,
        help="Whether the export data shape is dynamic or static.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["det", "rec", "cls"],
        help="Model type. Required when arg `is_dynamic_shape` is True.",
    )
    parser.add_argument(
        "--data_shape",
        type=int,
        nargs=2,
        help="The data shape [H, W] for exporting mindir files. Required when arg `is_dynamic_shape` is False. \
            It is recommended to be the same as the rescaled data shape in evaluation to get the best inference \
            performance.",
    )
    parser.add_argument(
        "--local_ckpt_path",
        type=str,
        help="Path to a local checkpoint. If set, export mindir by loading local ckpt. \
            Otherwise, export mindir by downloading online ckpt.",
    )
    parser.add_argument("--save_dir", type=str, default="", help="Directory to save the exported mindir file.")

    args = parser.parse_args()
    check_args(args)
    export(**vars(args))
