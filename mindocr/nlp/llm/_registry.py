"""llm registry and list"""

__all__ = [
    "list_llms",
    "is_llm",
    "llm_entrypoint",
    "list_llm_classes",
    "is_llm_class",
    "llm_class_entrypoint",
    "register_llm",
]

_llm_entrypoints = {}
_llm_class_entrypoints = {}


def register_llm(fn):
    # add llm to __all__ in module
    llm_name = fn.__name__
    # add entries to registry dict/sets
    _llm_entrypoints[llm_name] = fn

    return fn


def list_llms():
    all_llms = _llm_entrypoints.keys()

    return sorted(list(all_llms))


def is_llm(llm_name):
    """
    Check if a llm name exists
    """
    return llm_name in _llm_entrypoints


def llm_entrypoint(llm_name):
    """
    Fetch a llm entrypoint for specified llm name
    """
    return _llm_entrypoints[llm_name]


def register_llm_class(cls):
    # add llm to __all__ in module
    llm_class_name = cls.__name__
    # add entries to registry dict/sets
    _llm_class_entrypoints[llm_class_name] = cls

    return cls


def list_llm_classes():
    all_llm_classes = _llm_class_entrypoints.keys()

    return sorted(list(all_llm_classes))


def is_llm_class(llm_class_name):
    """
    Check if a llm name exists
    """
    return llm_class_name in _llm_class_entrypoints


def llm_class_entrypoint(llm_class_name):
    """
    Fetch a llm entrypoint for specified llm name
    """
    return _llm_class_entrypoints[llm_class_name]
