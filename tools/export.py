'''
Usage:
    To export all trained models from ckpt to mindir as listed in configs/, run

    python tools/export.py
'''
import sys
import os
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import mindspore as ms
from mindocr import list_models, build_model
import numpy as np


def export(name, task='rec'):
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')

    net = build_model(name, pretrained=True)
    net.set_train(False)

    if task=='rec':
        c, h, w = 3, 32, 100
    else:
        c, h, w = 3, 640, 640

    bs = 1
    x = ms.Tensor(np.ones([bs, c, h, w]), dtype=ms.float32)

    ms.export(net, x, file_name=name, file_format='MINDIR')


if __name__ == '__main__':
    names = list_models()
    for n in names:
        task = 'rec'
        if 'db' in n:
            task = 'det'
        export(n, task)




