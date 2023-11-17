"""backbone registry and list"""
import fnmatch

__all__ = [
    "list_backbones",
    "is_backbone",
    "backbone_entrypoint",
    "list_backbone_classes",
    "is_backbone_class",
    "backbone_class_entrypoint",
    "register_backbone",
]

_backbone_entrypoints = {}
_backbone_class_entrypoints = {}


def register_backbone(fn):

    # add backbone to __all__ in module
    backbone_name = fn.__name__
    '''
    if hasattr(mod, "__all__"):
        mod.__all__.append(backbone_name)
    else:
        mod.__all__ = [backbone_name]
    '''
    # add entries to registry dict/sets
    _backbone_entrypoints[backbone_name] = fn

    return fn


def list_backbones(filter='', exclude_filters=''):
    all_backbones = _backbone_entrypoints.keys()

    if filter:
        backbones = []
        include_filters = filter if isinstance(filter, (tuple, list)) else [filter]
        for f in include_filters:
            include_backbones = fnmatch.filter(all_backbones, f)  # include these backbones
            if include_backbones:
                backbones = set(backbones).union(include_backbones)
    else:
        backbones = all_backbones

    if exclude_filters:
        if not isinstance(exclude_filters, (tuple, list)):
            exclude_filters = [exclude_filters]
        for xf in exclude_filters:
            exclude_backbones = fnmatch.filter(backbones, xf)  # exclude these backbones
            if exclude_backbones:
                backbones = set(backbones).difference(exclude_backbones)

    backbones = sorted(list(backbones))

    return backbones


def is_backbone(backbone_name):
    """
    Check if a backbone name exists
    """
    return backbone_name in _backbone_entrypoints


def backbone_entrypoint(backbone_name):
    """
    Fetch a backbone entrypoint for specified backbone name
    """
    return _backbone_entrypoints[backbone_name]


def register_backbone_class(cls):

    # add backbone to __all__ in module
    backbone_class_name = cls.__name__
    # add entries to registry dict/sets
    _backbone_class_entrypoints[backbone_class_name] = cls

    return cls


def list_backbone_classes(filter='', exclude_filters=''):
    all_backbone_classes = _backbone_class_entrypoints.keys()

    return sorted(list(all_backbone_classes))


def is_backbone_class(backbone_class_name):
    """
    Check if a backbone name exists
    """
    return backbone_class_name in _backbone_class_entrypoints


def backbone_class_entrypoint(backbone_class_name):
    """
    Fetch a backbone entrypoint for specified backbone name
    """
    return _backbone_class_entrypoints[backbone_class_name]
