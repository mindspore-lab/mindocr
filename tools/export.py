"""
Export ckpt files to mindir files for inference.

Args:
    model_name (str): Name of the model to be converted.
    data_shape (int): The data shape [H, W] for exporting mindir files.
    local_ckpt_path (str): Path to a local checkpoint. If set, export mindir by loading local ckpt. Otherwise, export mindir by downloading online ckpt.
    save_dir (str): Directory to save the exported mindir file.

Example:
    >>> # Export mindir of model `dbnet_resnet50` by downloading online ckpt
    >>> python tools/export.py --model_name dbnet_resnet50 --data_shape 736 1280
    >>> # Export mindir of model `dbnet_resnet50` by loading local ckpt
    >>> python tools/export.py --model_name dbnet_resnet50 --data_shape 736 1280 --local_ckpt_path /path/to/local_ckpt

Notes:
    - Args `model_name` and `data_shape` are required to be specified when running export.py.
    - The `data_shape` is recommended to be the same as the rescaled data shape in evaluation to get the best inference performance.
    - The online ckpt files are downloaded from https://download.mindspore.cn/toolkits/mindocr/.
"""

import sys
import os

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import argparse
import mindspore as ms
from mindocr import list_models, build_model
import numpy as np


def export(name, data_shape, local_ckpt_path="", save_dir=""):
    ms.set_context(mode=ms.GRAPH_MODE) #, device_target='Ascend')
    if local_ckpt_path:
        net = build_model(name, pretrained=False, ckpt_load_path=local_ckpt_path)
    else:
        net = build_model(name, pretrained=True)
    net.set_train(False)

    h, w = data_shape
    bs, c = 1, 3
    x = ms.Tensor(np.ones([bs, c, h, w]), dtype=ms.float32)

    output_path = os.path.join(save_dir, name) + '.mindir'
    ms.export(net, x, file_name=output_path, file_format='MINDIR')

    print(f'=> Finish exporting {name} to {os.path.realpath(output_path)}. The data shape [H, W] is {data_shape}')


def check_args(args):
    if args.local_ckpt_path and not os.path.isfile(args.local_ckpt_path):
        raise ValueError(f"Local ckpt file {args.local_ckpt_path} does not exist. Please check arg 'local_ckpt_path'.")
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Convert model checkpoint to mindir format.")
    parser.add_argument(
        '--model_name',
        type=str,
        default="",
        required=True,
        choices=list_models(),
        help=f'Name of the model to be converted. Available choices: {list_models()}.')
    parser.add_argument(
        '--data_shape',
        type=int,
        nargs=2,
        default="",
        required=True,
        help=f'The data shape [H, W] for exporting mindir files. It is recommended to be the same as the rescaled data shape in evaluation to get the best inference performance.')
    parser.add_argument(
        '--local_ckpt_path',
        type=str,
        default="",
        help='Path to a local checkpoint. If set, export mindir by loading local ckpt. Otherwise, export mindir by downloading online ckpt.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default="",
        help='Directory to save the exported mindir file.')

    args = parser.parse_args()
    check_args(args)

    export(args.model_name, args.data_shape, args.local_ckpt_path, args.save_dir)
