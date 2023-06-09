import sys

sys.path.append(".")
import numpy as np
import pytest

import mindspore as ms

from mindocr import build_model, list_models
from tools.export import export


@pytest.mark.parametrize("model_name", ["dbnet_resnet50", "crnn_resnet34", "rare_resnet34"])
def test_mindir_infer(model_name):
    ms.set_context(mode=ms.GRAPH_MODE)

    task = "rec"
    if "db" in model_name or "east" in model_name or "pse" in model_name:
        task = "det"

    if task == "rec":
        c, h, w = 3, 32, 100
    else:
        c, h, w = 3, 736, 1280

    export(model_name, [h, w], local_ckpt_path="", save_dir="")

    fn = f"{model_name}.mindir"
    graph = ms.load(fn)
    model = ms.nn.GraphCell(graph)

    bs = 1
    x = ms.Tensor(np.ones([bs, c, h, w]), dtype=ms.float32)

    outputs_mindir = model(x)

    # get original ckpt outputs
    net = build_model(model_name, pretrained=True)
    outputs_ckpt = net(x)

    for i, o in enumerate(outputs_mindir):
        print("mindir net out: ", outputs_mindir[i].sum(), outputs_mindir[i].shape)
        print("ckpt net out: ", outputs_ckpt[i].sum(), outputs_ckpt[i].shape)
        assert float(outputs_mindir[i].sum().asnumpy()) == float(outputs_ckpt[i].sum().asnumpy())


if __name__ == "__main__":
    names = list_models()
    test_mindir_infer(names[0])
