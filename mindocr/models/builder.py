'''
build models
'''
from typing import Union
from ._registry import model_entrypoint, list_models, is_model
from .base_model import BaseModel

#def build_model(config: Union[dict,str]):
def build_model(config: Union[dict, str], **kwargs): #config: Union[dict,str]):
    '''
    There are two ways to build a model. 
        1. load a predefined model according the given model name. 
        2. build the model according to the detailed configuration of the each module (transform, backbone, neck and head), for lower-level architecture customization.

    Args:
        config: if it is a str, config is the model name. Predefined model with weights will be returned.
                if dict, config is a dictionary and the available keys are: 
                    model_name: string, model name in the registered models 
                    pretrained: bool, if True, download the pretrained weight for the preset url and load to the network.
                    backbone: dict, a dictionary containing the backbone config, the available keys are defined in backbones/builder.py 
                    neck: dict,  
                    head: dict,  
    
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

    return network
