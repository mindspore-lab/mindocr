'''
Usage:
    To export all trained models from ckpt to mindir as listed in configs/, run
       $  python tools/export.py

    To export a sepecific model, taking dbnet for example, run
       $ python tools/export.py --model_name dbnet_resnet50  --save_dir
'''
import sys
import os
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import argparse
import mindspore as ms
from mindocr import list_models, build_model
import numpy as np


def export(name, task='rec', pretrained=True, ckpt_load_path="", save_dir=""):
    ms.set_context(mode=ms.GRAPH_MODE) #, device_target='Ascend')

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Convert model checkpoint to mindir format.")
    parser.add_argument(
        '--model_name',
        type=str,
        default="",
        help='Name of the model to convert, choices: [crnn_resnet34, crnn_vgg7, dbnet_resnet50, ""]. You can lookup the name by calling mindocr.list_models(). If "", all models in list_models() will be converted.')
    parser.add_argument(
        '--pretrained',
        type=str2bool, nargs='?', const=True,
        default=True,
        help='Whether download and load the pretrained checkpoint into network.')
    parser.add_argument(
        '--ckpt_load_path',
        type=str,
        default="",
        help='Path to a local checkpoint. No need to set it if pretrained is True. If set, network weights will be loaded using this checkpoint file')
    parser.add_argument(
        '--save_dir',
        type=str,
        default="",
        help='Dir to save the exported model')

    args = parser.parse_args()
    mn = args.model_name

    if mn =="":
        names = list_models()
    else:
        names = [mn]

    for n in names:
        task = 'rec'
        if 'db' in n:
            task = 'det'
        export(n, task, args.pretrained, args.ckpt_load_path, args.save_dir)
