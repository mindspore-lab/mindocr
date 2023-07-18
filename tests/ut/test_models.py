import sys

sys.path.append(".")

import time

import numpy as np
import pytest
import yaml

import mindspore as ms

import mindocr
from mindocr.models import build_model

all_model_names = mindocr.list_models()
print("Registered models: ", all_model_names)

# yaml path and mode (0: graph, 1: pynative)
all_yamls = [
    ("configs/det/dbnet/db_r50_icdar15.yaml", [0, 1]),
    ("configs/rec/crnn/crnn_resnet34.yaml", [0, 1]),
    ("configs/rec/crnn/crnn_vgg7.yaml", [0, 1]),
    ("configs/rec/master/master_resnet31.yaml", [0]),
    ("configs/cls/mobilenetv3/cls_mv3.yaml", [0, 1]),
]
print("All config yamls: ", all_yamls)


def _infer_dummy(model, task="det", mode=0, verbose=True):
    print(task)

    bs = 8
    if task == "rec":
        c, h, w = 3, 32, 100
    elif task == "cls":
        c, h, w = 3, 48, 192
    else:
        c, h, w = 3, 640, 640
    print("net input shape: ", bs, c, h, w)
    x = ms.Tensor(np.random.rand(bs, c, h, w), dtype=ms.float32)
    ms.set_context(mode=mode)

    def predict(model, x):
        start = time.time()
        y = model(x)
        print(time.time() - start)
        return y

    def predict_parts(model, x):
        bout = model.backbone(x)
        print("backbone output feature shapes: ")
        for ftr in bout:
            print("\t", ftr.shape)
        nout = model.neck(bout)
        print("neck output shape: ", nout.shape)
        hout = model.head(nout)
        print("head output shape: ")
        if isinstance(hout, tuple):
            for ho in hout:
                print(ho.shape)
        else:
            print(hout.shape)

        return hout

    if verbose:
        out = predict_parts(model, x)
    else:
        out = predict(model, x)
    return out


@pytest.mark.parametrize("model_name", all_model_names)
@pytest.mark.parametrize("pretrained", [True, False])
def test_model_by_name(model_name, pretrained):
    print(model_name)
    build_model(model_name, pretrained=pretrained)
    # _infer_dummy(model)
    print("model created")


@pytest.mark.parametrize("yaml_and_modes", all_yamls)
def test_model_by_yaml(yaml_and_modes):
    print(yaml_and_modes)
    yaml_path, modes = yaml_and_modes
    with open(yaml_path) as fp:
        config = yaml.safe_load(fp)

    task = yaml_path.split("/")[1]

    model_config = config["model"]
    model = build_model(model_config)
    for mode in modes:
        _infer_dummy(model, task=task, mode=mode)


if __name__ == "__main__":
    print(all_model_names)
    test_model_by_name(all_model_names[2], True)
    # test_model_by_yaml(all_yamls[1])
