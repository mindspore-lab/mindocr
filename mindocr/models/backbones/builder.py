import importlib
from ._registry import backbone_entrypoint, is_backbone, backbone_class_entrypoint, is_backbone_class, list_backbones 

__all__ = ['build_backbone']

#support_backbones = _backbone_entrypoints.keys()
#support_backbone_classes = _backbone_class_entrypoints.keys() 

def build_backbone(name, **kwargs): #config: dict):
    '''
    Args:
        config (dict): config dict of the backbone including backbone name and hyper-params for the backbone.
            name:
            pretrained:
            other hyper-params for the backbone:

    Return:
        nn.Cell for backbone moduel

    Example:
        >>> # configure from model name
        >>> cfg = dict(name='det_resnet50', pretrained=True)
        >>> backbone = build_backbone(cfg) 
        >>> # 
        >>> cfg_from_class  = dict(name='DetResNet', layers=[3,4,6,3])
        >>> backbone = build_backbone(cfg_from_class) 
        >>> print(backbone)

    '''
    #name = config.pop('name')
    #kwargs = {k:v for k,v in config.items() if v is not None}

    if is_backbone(name):
        create_fn = backbone_entrypoint(name)
        backbone = create_fn(**kwargs)
    elif is_backbone_class(name):
        backbone_class = backbone_class_entrypoint(name)
        backbone = backbone_class(**kwargs)
        #TODO: load pretrained weights

    elif name[:7]=='mindcv.':
        # FIXME: it is not usable right now. 
        # TODO: update mindcv to get list of feature tensors, by adding feature_only parameter and out_indices to extract intermediate features. 
        mindcv_model_name = name[7:]
        import mindcv
        backbone = mindcv.create_model(mindcv_model_name, feature_only=True, **kwargs)
        backbone.construct = backbone.forward_features
    else:
        raise ValueError(f'Invalid backbone name: {name}, supported backbones are: {list_backbones()}')

    return backbone
