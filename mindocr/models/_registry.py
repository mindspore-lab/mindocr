'''model registry and list'''
import sys
import fnmatch
from collections import defaultdict

__all__ = [
    'list_models',
    'is_model',
    'model_entrypoint',
    'list_modules',
    'is_model_in_modules',
    'is_model_pretrained'
]

_module_to_models = defaultdict(set)
_model_to_module = {}
_model_entrypoints = {}
_model_has_pretrained = set()


def register_model(fn):
    # lookup containing module
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''

    # add model to __all__ in module
    model_name = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]

    # add entries to registry dict/sets
    _model_entrypoints[model_name] = fn
    _model_to_module[model_name] = module_name
    _module_to_models[module_name].add(model_name)
    has_pretrained = False
    if hasattr(mod, 'default_cfgs') and model_name in mod.default_cfgs:
        cfg = mod.default_cfgs[model_name]
        has_pretrained = 'url' in cfg and cfg['url']
    if has_pretrained:
        _model_has_pretrained.add(model_name)
    return fn

def list_models(filter='', module='', pretrained=False, exclude_filters=''):
    if module:
        all_models = list(_module_to_models[module])
    else:
        all_models = _model_entrypoints.keys()

    if filter:
        models = []
        include_filters = filter if isinstance(filter, (tuple, list)) else [filter]
        for f in include_filters:
            include_models = fnmatch.filter(all_models, f)  # include these models
            if include_models:
                models = set(models).union(include_models)
    else:
        models = all_models

    if exclude_filters:
        if not isinstance(exclude_filters, (tuple, list)):
            exclude_filters = [exclude_filters]
        for xf in exclude_filters:
            exclude_models = fnmatch.filter(models, xf)  # exclude these models
            if exclude_models:
                models = set(models).difference(exclude_models)

    if pretrained:
        models = _model_has_pretrained.intersection(models)

    models = sorted(list(models))

    return models


def is_model(model_name):
    """
    Check if a model name exists
    """
    return model_name in _model_entrypoints


def model_entrypoint(model_name):
    """
    Fetch a model entrypoint for specified model name
    """
    return _model_entrypoints[model_name]


def list_modules():
    """
    Return list of module names that contain models / model entrypoints
    """
    modules = _module_to_models.keys()
    return list(sorted(modules))


def is_model_in_modules(model_name, module_names):
    """
    Check if a model exists within a subset of modules
    Args:
        model_name (str) - name of model to check
        module_names (tuple, list, set) - names of modules to search in
    """
    assert isinstance(module_names, (tuple, list, set))
    return any(model_name in _module_to_models[n] for n in module_names)


def is_model_pretrained(model_name):
    return model_name in _model_has_pretrained
