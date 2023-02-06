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

def _infer_dummy(model):
    import mindspore as ms
    import time
    import numpy as np

    bs = 8
    x = ms.Tensor(np.random.rand(bs, 3, 640, 640), dtype=ms.float32)
    ms.set_context(mode=ms.PYNATIVE_MODE)

    def predict(model, x):
        start = time.time()
        y = model(x)
        print(time.time()-start)
        print(y.shape)
        return y

    out = predict(model, x)
    return out

@pytest.mark.parametrize('model_name', all_model_names)
def test_model_by_name(model_name):
    print(model_name)
    model = build_model(model_name, pretrained=False)
    _infer_dummy(model)
 

@pytest.mark.parametrize('yaml_fp', all_yamls)
def test_model_by_yaml(yaml_fp='configs/det/db_r50_icdar15.yaml'):
    print(yaml_fp)
    with open(yaml_fp) as fp:
        config  = yaml.safe_load(fp)    
    model_config = config['model']
    model = build_model(model_config)

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

    _infer_dummy(model)

    ''' TODO: check loading 
    ckpt_path = None   
    if ckpt_path is not None:
        param_dict = load_checkpoint(os.path.join(path, os.path.basename(default_cfg['url'])))
        load_param_into_net(model, param_dict)
    '''
  
if __name__ == '__main__':    
    #test_registry()
    #test_backbone()
    test_model_by_name('dbnet_r50')
    test_model_by_yaml('configs/det/db_r50_icdar15.yaml')
