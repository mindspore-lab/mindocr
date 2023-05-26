"""
Some utils while building models
"""
import collections.abc
import difflib
import logging
import os
from itertools import repeat
from typing import List, Optional

from mindspore import load_checkpoint, load_param_into_net

from .download import DownLoad, get_default_download_root


def get_checkpoint_download_root():
    return os.path.join(get_default_download_root(), "models")


class ConfigDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_pretrained(model, default_cfg, num_classes=1000, in_channels=3, filter_fn=None, auto_mapping=False):
    """load pretrained model depending on cfgs of model"""
    if "url" not in default_cfg or not default_cfg["url"]:
        logging.warning("Pretrained model URL is invalid")
        return

    # download files
    download_path = get_checkpoint_download_root()
    os.makedirs(download_path, exist_ok=True)
    DownLoad().download_url(default_cfg["url"], path=download_path)

    try:
        param_dict = load_checkpoint(os.path.join(download_path, os.path.basename(default_cfg["url"])))
    except:
        print(f'ERROR: Fails to load the checkpoint. Please check whether the checkpoint is downloaded successfully in {download_path} and is not zero-byte. You may try to manually download the checkpoint from ', default_cfg["url"])
        param_dict = dict()

    if auto_mapping:
        net_param = model.get_parameters()
        ckpt_param = list(param_dict.keys())
        remap = {}
        for param in net_param:
            if param.name not in ckpt_param:
                print('Cannot find a param to load: ', param.name)
                poss = difflib.get_close_matches(param.name, ckpt_param, n=3, cutoff=0.6)
                if len(poss) > 0:
                    print('=> Find most matched param: ', poss[0], ', loaded')
                    param_dict[param.name] = param_dict.pop(poss[0]) # replace
                    remap[param.name] = poss[0]
                else:
                    raise ValueError('Cannot find any matching param from: ', ckpt_param)

        if remap != {}:
            print('WARNING: Auto mapping succeed. Please check the found mapping names to ensure correctness')
            print('\tNet Param\t<---\tCkpt Param')
            for k in remap:
                print(f'\t{k}\t<---\t{remap[k]}')

    if in_channels == 1:
        conv1_name = default_cfg["first_conv"]
        logging.info("Converting first conv (%s) from 3 to 1 channel", conv1_name)
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

    print('INFO: Finish loading model checkpoint from: ', os.path.join(download_path, os.path.basename(default_cfg["url"])))


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
