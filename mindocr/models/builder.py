'''
build models
'''
from typing import Union
from mindspore import load_checkpoint, load_param_into_net
from ._registry import model_entrypoint, list_models, is_model
from .base_model import BaseModel

__all__ = ['build_model']

#def build_model(config: Union[dict,str]):
def build_model(config: Union[dict, str], **kwargs): #config: Union[dict,str]):
    '''
    There are two ways to build a model.
        1. load a predefined model according the given model name.
        2. build the model according to the detailed configuration of the each module (transform, backbone, neck and head), for lower-level architecture customization.

    Args:
        config (Union[dict, str]): if it is a str, config is the model name. Predefined model with weights will be returned.
                if dict, config is a dictionary and the available keys are:
                    model_name: string, model name in the registered models
                    pretrained: bool, if True, download the pretrained weight for the preset url and load to the network.
                    backbone: dict, a dictionary containing the backbone config, the available keys are defined in backbones/builder.py
                    neck: dict,
                    head: dict,
        kwargs: if config is a str of model name, kwargs contains the args for the model. 
    
    Return:
        nn.Cell

    Example:
    >>>  from mindocr.models import build_model
    >>>  net = build_model(cfg['model'])
    >>>  net = build_model(cfg['model'], ckpt_load_path='./r50_fpn_dbhead.ckpt') # build network and load checkpoint
    >>>  net = build_model('dbnet_r50', pretrained=True)

    '''
    if isinstance(config, str):
        # build model by specific model name
        model_name = config #config['name']
        if is_model(model_name):
            create_fn = model_entrypoint(model_name)
            '''
            kwargs = {}
            for k, v in config.items():
                if k!=model_name and v is not None:
                    kwargs[k] = v
            '''
            network = create_fn(**kwargs)
        else:
            raise ValueError(f'Invalid model name: {model_name}. Supported models are {list_models()}')

    elif isinstance(config, dict):
        # build model by given architecture config dict
        network = BaseModel(config)
    else:
        raise ValueError('Type error for config')
   
    # load checkpoint
    if 'ckpt_load_path' in kwargs:
        ckpt_path = kwargs['ckpt_load_path']
        assert ckpt_path not in ['', None], f'Please provide the correct \n`eval:\n\tckpt_load_path`\n in the yaml config file '
        print(f'INFO: Loading checkpoint from {ckpt_path}')
        params = load_checkpoint(ckpt_path)
        load_param_into_net(network, params)

    return network
