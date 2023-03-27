import sys
sys.path.append('.')

import yaml
import glob
import pytest
import numpy as np
import mindspore as ms
import mindocr
from mindocr.models.backbones import build_backbone
from mindocr.models import build_model
from mindspore import load_checkpoint, load_param_into_net

all_model_names = mindocr.list_models()
print('Registered models: ', all_model_names)

all_yamls = glob.glob('configs/*/*.yaml')
print('All config yamls: ', all_yamls)

def _infer_dummy(model, task='det', verbose=True):
    import mindspore as ms
    import time
    import numpy as np

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
        for k in hout:
            print('\r', k, hout[k].shape)

        return hout

    if verbose:
        out = predict_parts(model, x)
    else:
        out = predict(model, x)
    return out

@pytest.mark.parametrize('model_name', all_model_names)
def test_model_by_name(model_name):
    print(model_name)
    model = build_model(model_name, pretrained=False)
    _infer_dummy(model)


@pytest.mark.parametrize('yaml_fp', all_yamls)
def test_model_by_yaml(yaml_fp='configs/det/dbnet/db_r50_icdar15.yaml'):
    print(yaml_fp)
    with open(yaml_fp) as fp:
        config  = yaml.safe_load(fp)

    task = yaml_fp.split('/')[-2]

    if task == 'rec':
        from mindocr.postprocess.rec_postprocess import CTCLabelDecode
        dict_path = config['common']['character_dict_path']
        # read dict path and get class nums
        rec_info = CTCLabelDecode(character_dict_path=dict_path)
        num_classes = len(rec_info.character)
        config['model']['head']['out_channels'] = num_classes
        print('num characters: ', num_classes)

    model_config = config['model']
    model = build_model(model_config)
    _infer_dummy(model, task=task)

    '''
    model_config = {
            "backbone": {
                'name': 'det_resnet50',
                'pretrained': False
                },
            "neck": {
                "name": 'FPN',
                "out_channels": 256,
                },
            "head": {
                "name": 'ConvHead',
                "out_channels": 2,
                "k": 50
                }

            }
    '''

    ''' TODO: check loading
    ckpt_path = None
    if ckpt_path is not None:
        param_dict = load_checkpoint(os.path.join(path, os.path.basename(default_cfg['url'])))
        load_param_into_net(model, param_dict)
    '''

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='model config', add_help=False)
    parser.add_argument('-c', '--config', type=str, default='configs/det/dbnet/db_r50_icdar15.yaml',
                               help='YAML config file specifying default arguments (default='')')
    args = parser.parse_args()
    #test_registry()
    #test_backbone()
    #test_model_by_name('dbnet_r50')
    test_model_by_yaml(args.config)
