'''
The online ckpt files are downloaded from https://download.mindspore.cn/toolkits/mindocr/
Usage:
    To export all trained models from online ckpt to mindir as listed in configs/, run
       $ python tools/export.py

    To export a specific model by downloading online ckpt, taking dbnet_resnet50 for example, run
       $ python tools/export.py --model_name dbnet_resnet50

    To export a specific model by loading local ckpt, taking dbnet_resnet50 for example, run
       $ python tools/export.py --model_name dbnet_resnet50 --local_ckpt_path /path/to/local_ckpt
'''
import sys
import os
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import argparse
import mindspore as ms
from mindocr import list_models, build_model
import numpy as np


def export(name, task='rec', local_ckpt_path="", save_dir=""):
    ms.set_context(mode=ms.GRAPH_MODE) #, device_target='Ascend')
    if local_ckpt_path:
        net = build_model(name, pretrained=False, ckpt_load_path=local_ckpt_path)
    else:
        net = build_model(name, pretrained=True)
    net.set_train(False)

    # TODO: extend input shapes for more models
    if task=='rec':
        c, h, w = 3, 32, 100
    else:
        c, h, w = 3, 736, 1280

    bs = 1
    x = ms.Tensor(np.ones([bs, c, h, w]), dtype=ms.float32)

    output_path = os.path.join(save_dir, name) + '.mindir'
    ms.export(net, x, file_name=output_path, file_format='MINDIR')

    print(f'=> Finish exporting {name} to {output_path}')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "1"):
        return True
    elif v.lower() in ("no", "false", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def check_args(args):
    if args.local_ckpt_path:  # load local ckpt
        if not args.model_name:
            raise ValueError("Arg 'model_name' is empty. Please set 'model_name' if 'local_ckpt_path' is not None.")
        if args.model_name not in list_models():
            raise ValueError(f"Invalid 'model_name': {args.model_name}. 'model_name' must be one of names in {list_models()}.")
        if not os.path.isfile(args.local_ckpt_path):
            raise ValueError(f"No such ckpt file in this path: {args.local_ckpt_path}.")
    else:  # download online ckpt
        if args.model_name and args.model_name not in list_models():
            raise ValueError(f"Invalid 'model_name': {args.model_name}. 'model_name' must be empty or one of names in {list_models()}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Convert model checkpoint to mindir format.")
    parser.add_argument(
        '--model_name',
        type=str,
        default="",
        help='Name of the model to convert, choices: [crnn_resnet34, crnn_vgg7, dbnet_resnet50, ""]. You can lookup the name by calling mindocr.list_models(). If "", all models in list_models() will be converted.')
    parser.add_argument(
        '--local_ckpt_path',
        type=str,
        default="",
        help='Path to a local checkpoint. If set, export a specific model by loading local ckpt. Otherwise, export all models or a specific model by downloading online ckpt.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default="",
        help='Dir to save the exported model')

    args = parser.parse_args()
    check_args(args)

    if args.model_name == "":
        model_names = list_models()
    else:
        model_names = [args.model_name]

    for n in model_names:
        task = 'rec'
        if 'db' in n or 'east' in n:
            task = 'det'
        export(n, task, args.local_ckpt_path, args.save_dir)
