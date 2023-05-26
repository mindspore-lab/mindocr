from typing import Callable, Dict, List, Optional
import os
from mindspore import load_checkpoint, load_param_into_net
from ..backbones.mindcv_models.utils import load_pretrained


__all__ = ['load_model']


def load_model(network, load_from: str, filter_fn: Optional[Callable[[Dict], Dict]] = None):
    '''
    network: network
    load_from: can be url or local path to a checkpoint that will be loaded to the network.
    '''
    if load_from is not None:
        if load_from[:4] == 'http':
            url_cfg = {'url': load_from}
            load_pretrained(network, url_cfg, filter_fn=filter_fn)
        else:
            assert os.path.exists(load_from), f'Failed to load checkpoint. {load_from} NOT exist. \n Please check the path and set it in `eval-ckpt_load_path` or `model-pretrained` in the yaml config file '
            params = load_checkpoint(load_from)
            if filter_fn is not None:
                params = filter_fn(params)
            load_param_into_net(network, params)

            print(f'INFO: Finish loading model checkoint from {load_from}. If no parameter fail-load warning displayed, all checkpoint params have been successfully loaded.')
