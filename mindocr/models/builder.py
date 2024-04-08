"""
build models
"""
from typing import Union

from mindspore.amp import auto_mixed_precision

from ._registry import is_model, list_models, model_entrypoint
from .base_model import BaseModel
from .utils import load_model, set_amp_attr

__all__ = ["build_model"]


def build_model(name_or_config: Union[str, dict], **kwargs):
    """
    There are two ways to build a model.
        1. load a predefined model according the given model name.
        2. build the model according to the detailed configuration of the each module (transform, backbone, neck and
        head), for lower-level architecture customization.

    Args:
        name_or_config (Union[dict, str]): model name or config
            if it's a string, it should be a model name (which can be found by mindocr.list_models())
            if it's a dict, it should be an architecture configuration defining the backbone/neck/head components
            (e.g., parsed from yaml config).

        kwargs (dict): options
            if name_or_config is a model name, supported args in kwargs are:
                - pretrained (bool): if True, pretrained checkpoint will be downloaded and loaded into the network.
                - ckpt_load_path (str): path to checkpoint file. if a non-empty string given, the local checkpoint will
                  loaded into the network.
            if name_or_config is an architecture definition dict, supported args are:
                - ckpt_load_path (str): path to checkpoint file.

    Return:
        nn.Cell

    Example:
    >>>  from mindocr.models import build_model
    >>>  net = build_model(cfg['model'])
    >>>  net = build_model(cfg['model'], ckpt_load_path='./r50_fpn_dbhead.ckpt') # build network and load checkpoint
    >>>  net = build_model('dbnet_resnet50', pretrained=True)

    """
    is_customized_model = True
    if isinstance(name_or_config, str):
        # build model by specific model name
        model_name = name_or_config
        if is_model(model_name):
            create_fn = model_entrypoint(model_name)
            network = create_fn(**kwargs)
        else:
            raise ValueError(
                f"Invalid model name: {model_name}. Supported models are {list_models()}"
            )
        is_customized_model = False
    elif isinstance(name_or_config, dict):
        network = BaseModel(name_or_config)
    else:
        raise ValueError("Type error for config")

    # load checkpoint
    if "ckpt_load_path" in kwargs:
        load_from = kwargs["ckpt_load_path"]
        if isinstance(load_from, bool) and is_customized_model:
            raise ValueError(
                "Cannot find the pretrained checkpoint for a customized model without giving the url or local path "
                "to the checkpoint.\nPlease specify the url or local path by setting `model-pretrained` (if training) "
                "or `eval-ckpt_load_path` (if evaluation) in the yaml config"
            )

        load_model(network, load_from)

    if "amp_level" in kwargs:
        auto_mixed_precision(network, amp_level=kwargs["amp_level"])
        set_amp_attr(network, kwargs["amp_level"])

    return network
