"""
Some utils while building models
"""
import collections.abc
import difflib
import logging
import os
from copy import deepcopy
from itertools import repeat
from typing import List, Optional

from mindspore import load_checkpoint, load_param_into_net

from .download import DownLoad, get_default_download_root

_logger = logging.getLogger(__name__)


def get_checkpoint_download_root():
    return os.path.join(get_default_download_root(), "models")


class ConfigDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def download_pretrained(default_cfg):
    """Download the pretrained ckpt from url to local path"""
    if "url" not in default_cfg or not default_cfg["url"]:
        _logger.warning("Pretrained model URL is invalid")
        return

    # download files
    download_path = get_checkpoint_download_root()
    os.makedirs(download_path, exist_ok=True)
    file_path = DownLoad().download_url(default_cfg["url"], path=download_path)
    return file_path


def auto_map(model, param_dict):
    """Raname part of the param_dict such that names from checkpoint and model are consistent"""
    updated_param_dict = deepcopy(param_dict)
    net_param = model.get_parameters()
    ckpt_param = list(updated_param_dict.keys())
    remap = {}
    for param in net_param:
        if param.name not in ckpt_param:
            _logger.info(f'Cannot find a param to load: {param.name}')
            poss = difflib.get_close_matches(param.name, ckpt_param, n=3, cutoff=0.6)
            if len(poss) > 0:
                _logger.info(f'=> Find most matched param: {poss[0]},  loaded')
                updated_param_dict[param.name] = updated_param_dict.pop(poss[0])  # replace
                remap[param.name] = poss[0]
            else:
                raise ValueError('Cannot find any matching param from: ', ckpt_param)

    if remap != {}:
        _logger.warning('Auto mapping succeed. Please check the found mapping names to ensure correctness')
        _logger.info('\tNet Param\t<---\tCkpt Param')
        for k in remap:
            _logger.info(f'\t{k}\t<---\t{remap[k]}')
    return updated_param_dict


def load_pretrained(model, default_cfg, num_classes=1000, in_channels=3, filter_fn=None, auto_mapping=False):
    """load pretrained model depending on cfgs of model"""
    file_path = download_pretrained(default_cfg)

    try:
        param_dict = load_checkpoint(file_path)
    except Exception:
        _logger.error(
            f"Fails to load the checkpoint. Please check whether the checkpoint is downloaded successfully"
            f"as `{file_path}` and is not zero-byte. You may try to manually download the checkpoint "
            f"from {default_cfg['url']}"
        )
        param_dict = dict()

    if auto_mapping:
        param_dict = auto_map(model, param_dict)

    if in_channels == 1:
        conv1_name = default_cfg["first_conv"]
        _logger.info(f"Converting first conv {conv1_name} from 3 to 1 channel")
        con1_weight = param_dict[conv1_name + ".weight"]
        con1_weight.set_data(con1_weight.sum(axis=1, keepdims=True), slice_shape=True)
    elif in_channels != 3:
        raise ValueError("Invalid in_channels for pretrained weights")

    if 'classifier' in default_cfg:
        classifier_name = default_cfg["classifier"]
        if num_classes == 1000 and default_cfg["num_classes"] == 1001:
            classifier_weight = param_dict[classifier_name + ".weight"]
            classifier_weight.set_data(classifier_weight[:1000], slice_shape=True)
            classifier_bias = param_dict[classifier_name + ".bias"]
            classifier_bias.set_data(classifier_bias[:1000], slice_shape=True)
        elif num_classes != default_cfg["num_classes"]:
            params_names = list(param_dict.keys())
            param_dict.pop(
                _search_param_name(params_names, classifier_name + ".weight"),
                "No Parameter {} in ParamDict".format(classifier_name + ".weight"),
            )
            param_dict.pop(
                _search_param_name(params_names, classifier_name + ".bias"),
                "No Parameter {} in ParamDict".format(classifier_name + ".bias"),
            )

    if filter_fn is not None:
        param_dict = filter_fn(param_dict)

    load_param_into_net(model, param_dict)

    _logger.info(f'Finish loading model checkpoint from: {file_path}')


def make_divisible(
    v: float,
    divisor: int,
    min_value: Optional[int] = None,
) -> int:
    """Find the smallest integer larger than v and divisible by divisor."""
    if not min_value:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


def _search_param_name(params_names: List, param_name: str) -> str:
    for pi in params_names:
        if param_name in pi:
            return pi
    return ""
