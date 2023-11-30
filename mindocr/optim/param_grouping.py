"""
group parameters for setting different weight decay or learning rate for different layers in the network.
"""
import logging
from copy import deepcopy
from typing import Iterable

import mindspore as ms
from mindspore.nn.learning_rate_schedule import LearningRateSchedule

__all__ = ["create_group_params"]

supported_grouping_strategies = ["svtr", "filter_norm_and_bias", "visionlan"]
_logger = logging.getLogger(__name__)


def grouping_default(params, weight_decay):
    decay_params = []
    no_decay_params = []

    filter_keys = ["beta", "gamma", "bias"]
    for param in params:
        filter_param = False
        for k in filter_keys:
            if k in param.name:
                filter_param = True

        if filter_param:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params},
        {"order_params": params},
    ]


def grouping_svtr(params, weight_decay):
    decay_params = []
    no_decay_params = []

    filter_keys = ["norm", "pos_embed"]

    for param in params:
        filter_param = False
        for k in filter_keys:
            # also filter the one dimensional parameters
            if k in param.name or len(param.shape) == 1:
                filter_param = True
                break

        if filter_param:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params},
        {"order_params": params},
    ]


def grouping_visionlan(params, weight_decay, learning_rate, training_step):
    if training_step == "LF_1" or training_step == "LA":
        return params
    elif training_step == "LF_2":
        # TODO: supports model parallelism in the future
        mlm = []
        pre_mlm_pp = []
        pre_mlm_w = []
        for param in params:
            if "MLM_VRM" in param.name:
                if "MLM_VRM.MLM" in param.name:
                    # model.head.MLM_VRM.MLM.parameters()
                    mlm.append(param)
                elif "MLM_VRM.Prediction.pp_share" in param.name:
                    # model.head.MLM_VRM.Prediction.pp_share.parameters()
                    pre_mlm_pp.append(param)
                elif "MLM_VRM.Prediction.w_share" in param.name:
                    # model.head.MLM_VRM.Prediction.w_share.parameters()
                    pre_mlm_w.append(param)

        total_ids = []
        for param_grps in [mlm, pre_mlm_pp, pre_mlm_w]:
            for param in param_grps:
                total_ids.append(id(param))
        group_base_params = [p for p in params if id(p) in total_ids]
        group_small_params = [p for p in params if id(p) not in total_ids]
        base_lr = learning_rate
        if isinstance(learning_rate, float) or isinstance(learning_rate, ms.Tensor):
            small_lr = base_lr * 0.1
        elif isinstance(learning_rate, LearningRateSchedule):
            small_lr = deepcopy(learning_rate)
            small_lr.learning_rate = small_lr.learning_rate * 0.1
        elif isinstance(learning_rate, Iterable):
            # Iterable
            small_lr = [x * 0.1 for x in base_lr]
        return [
            {"params": group_base_params, "lr": base_lr, "weight_decay": weight_decay},
            {"params": group_small_params, "lr": small_lr, "weight_decay": weight_decay},
        ]
    else:
        raise ValueError(f"incorrect trainig step {training_step}")


def create_group_params(params, weight_decay=0, grouping_strategy=None, no_weight_decay_params=[], **kwargs):
    """
    create group parameters for setting different weight decay or learning rate for different layers in the network.

    Args:
        params: network params
        weight_decay (float): weight decay value
        grouping_strategy (str): name of the hard-coded grouping strategy. If not None, group parameters according to
            the predefined function and `no_weight_decay_params` will not make effect.
        no_weight_decay_params (list): list of the param name substrings that will be picked to exclude from
            weight decay. If a parameter containing one of the substrings in the list, the parameter will not be applied
            with weight decay. (Tips: param names can be checked by `[p.name for p in network.trainable_params()]`

    Return:
        list[dict], grouped parameters
    """

    # TODO: assert valid arg names
    gp = grouping_strategy

    if gp is not None:
        if weight_decay == 0:
            _logger.warning("weight decay is 0 in param grouping, which is meaningless. Please check config setting.")
        if len(no_weight_decay_params) > 0:
            _logger.warning(
                "Both grouping_strategy and no_weight_decay_params are set, but grouping_strategy is of prior."
                " no_weight_decay_params={no_weight_decay_params} will not make effect."
            )

        if gp == "svtr":
            return grouping_svtr(params, weight_decay)
        elif gp == "visionlan":
            assert (
                "lr" in kwargs and "training_step" in kwargs
            ), "expect to have lr and training step to create group params "
            return grouping_visionlan(params, weight_decay, kwargs["lr"], kwargs["training_step"])
        elif gp == "filter_norm_and_bias":
            return grouping_default(params, weight_decay)
        else:
            raise ValueError(
                f"The grouping function for {gp} is not defined. "
                f"Valid grouping strategies are {supported_grouping_strategies}"
            )

    elif len(no_weight_decay_params) > 0:
        assert weight_decay > 0, f"Invalid weight decay value {weight_decay} for param grouping."
        decay_params = []
        no_decay_params = []
        for param in params:
            filter_param = False
            for k in no_weight_decay_params:
                if k in param.name:
                    filter_param = True

            if filter_param:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params},
            {"order_params": params},
        ]
    else:
        _logger.info("no parameter grouping is applied.")
        return params
