import sys
sys.path.append('.')

import time
import yaml
import pytest
import numpy as np
import mindspore as ms
import mindocr
from mindocr.models import build_model

all_model_names = mindocr.list_models()
print('Registered models: ', all_model_names)

all_yamls = ['configs/det/dbnet/db_r50_icdar15.yaml',
             'configs/rec/crnn/crnn_resnet34.yaml',
             'configs/rec/crnn/crnn_vgg7.yaml']
print('All config yamls: ', all_yamls)

def _infer_dummy(model, task='det', verbose=True):
    print(task)

    bs = 8
    if task=='rec':
        c, h, w = 3, 32, 100
    else:
        c, h, w = 3, 640, 640
    print(f'net input shape: ', bs, c, h, w)
    x = ms.Tensor(np.random.rand(bs, c, h, w), dtype=ms.float32)
    ms.set_context(mode=ms.PYNATIVE_MODE)

    def predict(model, x):
        start = time.time()
        y = model(x)
        print(time.time()-start)
        return y

    def predict_parts(model, x):
        bout = model.backbone(x)
        print('backbone output feature shapes: ')
        for ftr in bout:
            print('\t', ftr.shape)
        nout = model.neck(bout)
        print('neck output shape: ', nout.shape)
        hout = model.head(nout)
        print('head output shape: ')
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

@pytest.mark.parametrize('model_name', all_model_names)
@pytest.mark.parametrize('pretrained', [True, False])
def test_model_by_name(model_name, pretrained):
    print(model_name)
    model = build_model(model_name, pretrained=pretrained)
    #_infer_dummy(model)
    print("model created")

@pytest.mark.parametrize('yaml_fp', all_yamls)
def test_model_by_yaml(yaml_fp):
    print(yaml_fp)
    with open(yaml_fp) as fp:
        config  = yaml.safe_load(fp)

    task = yaml_fp.split('/')[1]

    if task == 'rec':
        from mindocr.postprocess.rec_postprocess import RecCTCLabelDecode
        dict_path = config['common']['character_dict_path']
        # read dict path and get class nums
        rec_info = RecCTCLabelDecode(character_dict_path=dict_path)
        num_classes = len(rec_info.character)
        config['model']['head']['out_channels'] = num_classes
        print('num characters: ', num_classes)

    model_config = config['model']
    model = build_model(model_config)
    _infer_dummy(model, task=task)


if __name__ == '__main__':
    print(all_model_names)
    test_model_by_name(all_model_names[2], True)
    #test_model_by_yaml(all_yamls[1])
