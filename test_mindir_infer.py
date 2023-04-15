import mindspore as ms
import numpy as np
from mindocr import list_models, build_model


def test_mindir_infer(name, task='rec'):
    fn = f"{name}.mindir"

    ms.set_context(mode=ms.GRAPH_MODE)
    graph = ms.load(fn)
    model = ms.nn.GraphCell(graph)

    task = 'rec'
    if 'db' in fn:
        task = 'det'

    if task=='rec':
        c, h, w = 3, 32, 100
    else:
        c, h, w = 3, 640, 640

    bs = 1
    x = ms.Tensor(np.ones([bs, c, h, w]), dtype=ms.float32)

    outputs_mindir = model(x)

    # get original ckpt outputs
    net = build_model(name, pretrained=True)
    outputs_ckpt = net(x)

    for i, o in enumerate(outputs_mindir):
        print('mindir net out: ', outputs_mindir[i].sum(), outputs_mindir[i].shape)
        print('ckpt net out: ', outputs_ckpt[i].sum(), outputs_mindir[i].shape)
        assert outputs_mindir[i].sum()==outputs_ckpt[i].sum()


if __name__ == '__main__':
    names = list_models()
    for n in names:
        task = 'rec'
        if 'db' in n:
            task = 'det'
        print(n)
        test_mindir_infer(n, task)




