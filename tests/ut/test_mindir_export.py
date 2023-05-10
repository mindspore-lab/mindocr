import sys
sys.path.append('.')
import mindspore as ms
import pytest
import numpy as np
from mindocr import list_models, build_model
from tools.export import export

@pytest.mark.parametrize('name', ['dbnet_resnet50', 'crnn_resnet34'])
def test_mindir_infer(name):
    task = 'rec'
    if 'db' in name:
        task = 'det'

    export(name, task)

    fn = f"{name}.mindir"

    ms.set_context(mode=ms.GRAPH_MODE)
    graph = ms.load(fn)
    model = ms.nn.GraphCell(graph)

    if task=='rec':
        c, h, w = 3, 32, 100
    else:
        c, h, w = 3, 736, 1280

    bs = 1
    x = ms.Tensor(np.ones([bs, c, h, w]), dtype=ms.float32)

    outputs_mindir = model(x)

    # get original ckpt outputs
    net = build_model(name, pretrained=True)
    outputs_ckpt = net(x)

    for i, o in enumerate(outputs_mindir):
        print('mindir net out: ', outputs_mindir[i].sum(), outputs_mindir[i].shape)
        print('ckpt net out: ', outputs_ckpt[i].sum(), outputs_mindir[i].shape)
        assert float(outputs_mindir[i].sum().asnumpy())==float(outputs_ckpt[i].sum().asnumpy())


if __name__ == '__main__':
    names = list_models()
    test_mindir_infer(names[0])

