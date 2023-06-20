import os
from typing import Callable, Dict, Optional

from mindspore import load_checkpoint, load_param_into_net

from mindocr.utils.logger import Logger

from ..backbones.mindcv_models.utils import auto_map, download_pretrained

__all__ = ["load_model", "drop_inconsistent_shape_parameters"]
_logger = Logger("mindocr")


def drop_inconsistent_shape_parameters(model, param_dict):
    updated_param_dict = dict()
    for param in model.get_parameters():
        name = param.name
        if name in param_dict:
            if param_dict[name].shape == param.shape:
                updated_param_dict[name] = param_dict[name]
            else:
                _logger.warning(
                    f"Dropping checkpoint parameter `{name}` with shape `{param_dict[name].shape}`, "
                    f"which is inconsistent with cell shape `{param.shape}`"
                )
        else:
            _logger.warning(f"Cannot find checkpoint parameter `{name}`.")
    return updated_param_dict


def load_model(
    network,
    load_from: Optional[str] = None,
    filter_fn: Optional[Callable[[Dict], Dict]] = None,
    auto_mapping: bool = False,
    strict: bool = False,
):
    """
    Load the checkpoint into the model

    Args:
        network: network
        load_from: a string that can be url or local path to a checkpoint, that will be loaded to the network.
        filter_fn: a function filtering the parameters that will be loading into the network. If it is None,
            all parameters will be loaded.
        auto_mapping: when it is True, then load the paramters even if the names are slightly different
        strict: If it is true, then the shape and the type of the parameters in the checkpoint and the network
            should be consistent
            raise exception if they do not match.
    """
    if load_from is None:
        return

    if load_from[:4] == "http":
        url_cfg = {"url": load_from}
        local_ckpt_path = download_pretrained(url_cfg)
    else:
        local_ckpt_path = load_from

    assert local_ckpt_path and os.path.exists(local_ckpt_path), (
        f"Failed to load checkpoint. `{local_ckpt_path}` NOT exist. \n"
        "Please check the path and set it in `eval-ckpt_load_path` or `model-pretrained` in the yaml config file "
    )

    params = load_checkpoint(local_ckpt_path)

    if filter_fn is not None:
        params = filter_fn(params)

    if auto_mapping:
        params = auto_map(network, params)

    if not strict:
        params = drop_inconsistent_shape_parameters(network, params)

    load_param_into_net(network, params, strict_load=strict)

    _logger.info(
        f"Finish loading model checkoint from {load_from}. "
        "If no parameter fail-load warning displayed, all checkpoint params have been successfully loaded."
    )
