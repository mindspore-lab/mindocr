import os
import functools
from typing import Optional
import warnings
import inspect

from mindspore.nn import Cell


def _get_user_home():
    return os.path.expanduser("~")


def _get_mindocr_home():
    if "MINDOCR_HOME" in os.environ:
        home_path = os.environ["MINDOCR_HOME"]
        if os.path.exists(home_path):
            if os.path.isdir(home_path):
                return home_path
            else:
                raise RuntimeError("The environment variable MINDOCR_HOME {} is not a directory.".format(home_path))
        else:
            return home_path
    return os.path.join(_get_user_home(), ".mindocr")


def _get_sub_home(directory, parent_home=_get_mindocr_home()):
    home = os.path.join(parent_home, directory)
    if not os.path.exists(home):
        os.makedirs(home, exist_ok=True)
    return home


MODEL_HOME = _get_sub_home("models")


class InitTrackerMeta(type(Cell)):
    """
    This metaclass wraps the `__init__` method of a class to add `init_config`
    attribute for instances of that class, and `init_config` use a dict to track
    the initial configuration. If the class has `_pre_init` or `_post_init`
    method, it would be hooked before or after `__init__` and called as
    `_pre_init(self, init_fn, init_args)` or `_post_init(self, init_fn, init_args)`.
    Since InitTrackerMeta would be used as metaclass for pretrained model classes,
    which always are Layer and `type(Layer)` is not `type`, thus use `type(Layer)`
    rather than `type` as base class for it to avoid inheritance metaclass
    conflicts.
    """

    def __init__(cls, name, bases, attrs):
        init_func = cls.__init__
        # If attrs has `__init__`, wrap it using accessable `_pre_init, _post_init`.
        # Otherwise, no need to wrap again since the super cls has been wraped.
        # TODO: remove reduplicated tracker if using super cls `__init__`
        pre_init_func = getattr(cls, "_pre_init", None) if "__init__" in attrs else None
        post_init_func = getattr(cls, "_post_init", None) if "__init__" in attrs else None
        cls.__init__ = InitTrackerMeta.init_and_track_conf(init_func, pre_init_func, post_init_func)
        super(InitTrackerMeta, cls).__init__(name, bases, attrs)

    @staticmethod
    def init_and_track_conf(init_func, pre_init_func=None, post_init_func=None):
        """
        wraps `init_func` which is `__init__` method of a class to add `init_config`
        attribute for instances of that class.
        Args:
            init_func (callable): It should be the `__init__` method of a class.
                warning: `self` always is the class type of down-stream model, eg: BertForTokenClassification
            pre_init_func (callable, optional): If provided, it would be hooked after
                `init_func` and called as `pre_init_func(self, init_func, *init_args, **init_args)`.
                Default None.
            post_init_func (callable, optional): If provided, it would be hooked after
                `init_func` and called as `post_init_func(self, init_func, *init_args, **init_args)`.
                Default None.

        Returns:
            function: the wrapped function
        """

        @functools.wraps(init_func)
        def __impl__(self, *args, **kwargs):
            # registed helper by `pre_init_func`
            if pre_init_func:
                pre_init_func(self, init_func, *args, **kwargs)
            # keep full configuration
            init_func(self, *args, **kwargs)
            # registed helper by `post_init_func`
            if post_init_func:
                post_init_func(self, init_func, *args, **kwargs)
            self.init_config = kwargs
            if args:
                kwargs["init_args"] = args
            kwargs["init_class"] = self.__class__.__name__

        return __impl__

    def __setattr__(self, name, value):
        value = adapt_stale_fwd_patch(self, name, value)
        return super(InitTrackerMeta, self).__setattr__(name, value)


def adapt_stale_fwd_patch(self, name, value):
    """
    Since there are some monkey patches for forward of PretrainedModel, such as
    model compression, we make these patches compatible with the latest forward
    method.
    """
    if name == "forward":
        # NOTE(guosheng): In dygraph to static, `layer.forward` would be patched
        # by an instance of `StaticFunction`. And use string compare to avoid to
        # import fluid.
        if type(value).__name__.endswith("StaticFunction"):
            return value
        if hasattr(inspect, "getfullargspec"):
            (
                patch_spec_args,
                patch_spec_varargs,
                patch_spec_varkw,
                patch_spec_defaults,
                _,
                _,
                _,
            ) = inspect.getfullargspec(value)
            (spec_args, spec_varargs, spec_varkw, spec_defaults, _, _, _) = inspect.getfullargspec(self.forward)
        else:
            (patch_spec_args, patch_spec_varargs, patch_spec_varkw, patch_spec_defaults) = inspect.getargspec(value)
            (spec_args, spec_varargs, spec_varkw, spec_defaults) = inspect.getargspec(self.forward)
        new_args = [
            arg
            for arg in ("output_hidden_states", "output_attentions", "return_dict")
            if arg not in patch_spec_args and arg in spec_args
        ]

        if new_args:
            if self.__module__.startswith("paddlenlp"):
                warnings.warn(
                    f"The `forward` method of {self.__class__ if isinstance(self, Cell) else self} is patched and the patch "
                    "might be based on an old oversion which missing some "
                    f"arguments compared with the latest, such as {new_args}. "
                    "We automatically add compatibility on the patch for "
                    "these arguemnts, and maybe the patch should be updated."
                )
            else:
                warnings.warn(
                    f"The `forward` method of {self.__class__ if isinstance(self, Cell) else self} "
                    "is patched and the patch might be conflict with patches made "
                    f"by paddlenlp which seems have more arguments such as {new_args}. "
                    "We automatically add compatibility on the patch for "
                    "these arguemnts, and maybe the patch should be updated."
                )
            if isinstance(self, Cell) and inspect.isfunction(value):

                @functools.wraps(value)
                def wrap_fwd(*args, **kwargs):
                    for arg in new_args:
                        kwargs.pop(arg, None)
                    return value(self, *args, **kwargs)

            else:

                @functools.wraps(value)
                def wrap_fwd(*args, **kwargs):
                    for arg in new_args:
                        kwargs.pop(arg, None)
                    return value(*args, **kwargs)

            return wrap_fwd
    return value


def resolve_cache_dir(pretrained_model_name_or_path: str, from_hf_hub: bool, cache_dir: Optional[str] = None) -> str:
    """resolve cache dir for PretrainedModel and PretrainedConfig

    Args:
        pretrained_model_name_or_path (str): the name or path of pretrained model
        from_hf_hub (bool): if load from huggingface hub
        cache_dir (str): cache_dir for models
    """
    if os.path.isdir(pretrained_model_name_or_path):
        return pretrained_model_name_or_path

    # hf hub library takes care of appending the model name so we don't append the model name
    if from_hf_hub:
        if cache_dir is not None:
            return cache_dir
        else:
            return os.environ.get("HUGGINGFACE_HUB_CACHE", MODEL_HOME)  # TODO: from paddlenlp.utils.env import HF_CACHE_HOME, MODEL_HOME
    else:
        if cache_dir is not None:
            # since model_clas.from_pretrained calls config_clas.from_pretrained, the model_name may get appended twice
            if cache_dir.endswith(pretrained_model_name_or_path):
                return cache_dir
            else:
                return os.path.join(cache_dir, pretrained_model_name_or_path)
        return os.path.join(MODEL_HOME, pretrained_model_name_or_path)  # TODO: from paddlenlp.utils.env import MODEL_HOME


def fn_args_to_dict(func, *args, **kwargs):
    """
    Inspect function `func` and its arguments for running, and extract a
    dict mapping between argument names and keys.
    """
    if hasattr(inspect, "getfullargspec"):
        (spec_args, spec_varargs, spec_varkw, spec_defaults, _, _, _) = inspect.getfullargspec(func)
    else:
        (spec_args, spec_varargs, spec_varkw, spec_defaults) = inspect.getargspec(func)
    # add positional argument values
    init_dict = dict(zip(spec_args, args))
    # add default argument values
    kwargs_dict = dict(zip(spec_args[-len(spec_defaults) :], spec_defaults)) if spec_defaults else {}
    for k in list(kwargs_dict.keys()):
        if k in init_dict:
            kwargs_dict.pop(k)
    kwargs_dict.update(kwargs)
    init_dict.update(kwargs_dict)
    return init_dict
