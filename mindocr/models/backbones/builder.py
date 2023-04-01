import importlib
from ._registry import backbone_entrypoint, is_backbone, backbone_class_entrypoint, is_backbone_class, list_backbones
from .mindcv_wrapper import MindCVBackboneWrapper

__all__ = ['build_backbone']

def build_backbone(name, **kwargs):
    '''
    Build the backbone network.

    Args:
        name (str): the backbone name, which can be a registered backbone class name
                        or a registered backbone (function) name.
        kwargs (dict): input args for the backbone
           1) if `name` is in the registered backbones (e.g. det_resnet50), kwargs include args for backbone creating likes `pretrained`
           2) if `name` is in the registered backbones class (e.g. DetResNet50), kwargs include args for the backbone configuration like `layers`.

    Return:
        nn.Cell for backbone module

    Construct:
        Input: Tensor
        Output: List[Tensor]

    Example:
        >>> # build using backbone function name
        >>> from mindocr.models.backbones import build_backbone
        >>> backbone = build_backbone('det_resnet50', pretrained=True)
        >>> # build using backbone class name
        >>> from mindocr.models.backbones.mindcv_models.resnet import Bottleneck
        >>> cfg_from_class = dict(name='DetResNet', Bottleneck, layers=[3,4,6,3])
        >>> backbone = build_backbone(**cfg_from_class)
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

    elif 'mindcv' in name:
        # TODO: update mindcv to get list of feature tensors, by adding feature_only parameter and out_indices to extract intermediate features.
        backbone = MindCVBackboneWrapper(name, **kwargs)
    else:
        raise ValueError(f'Invalid backbone name: {name}, supported backbones are: {list_backbones()}')

    return backbone
