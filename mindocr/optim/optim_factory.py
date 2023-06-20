""" optim factory """
import inspect
import os
from typing import Optional

from mindspore import load_checkpoint, load_param_into_net, nn

from ..utils.logger import Logger
from .adamw import AdamW
from .adan import Adan
from .lion import Lion
from .nadam import NAdam

__all__ = ["create_optimizer"]
_logger = Logger("mindocr")


def init_group_params(params, weight_decay):
    decay_params = []
    no_decay_params = []

    for param in params:
        if "beta" not in param.name and "gamma" not in param.name and "bias" not in param.name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params},
        {"order_params": params},
    ]


def create_optimizer(
    params,
    opt: str = "adam",
    lr: Optional[float] = 1e-3,
    weight_decay: float = 0,
    momentum: float = 0.9,
    nesterov: bool = False,
    filter_bias_and_bn: bool = True,
    loss_scale: float = 1.0,
    schedule_decay: float = 4e-3,
    checkpoint_path: str = "",
    eps: float = 1e-10,
    **kwargs,
):
    r"""Creates optimizer by name.

    Args:
        params: network parameters. Union[list[Parameter],list[dict]], which must be the list of parameters
            or list of dicts. When the list element is a dictionary, the key of the dictionary can be
            "params", "lr", "weight_decay","grad_centralization" and "order_params".
        opt: wrapped optimizer. You could choose like 'sgd', 'nesterov', 'momentum', 'adam', 'adamw', 'lion',
            'rmsprop', 'adagrad', 'lamb'. 'adam' is the default choose for convolution-based networks.
            'adamw' is recommended for ViT-based networks. Default: 'adam'.
        lr: learning rate: float or lr scheduler. Fixed and dynamic learning rate are supported. Default: 1e-3.
        weight_decay: weight decay factor. It should be noted that weight decay can be a constant value or a Cell.
            It is a Cell only when dynamic weight decay is applied. Dynamic weight decay is similar to
            dynamic learning rate, users need to customize a weight decay schedule only with global step as input,
            and during training, the optimizer calls the instance of WeightDecaySchedule to get the weight decay value
            of current step. Default: 0.
        momentum: momentum if the optimizer supports. Default: 0.9.
        nesterov: Whether to use Nesterov Accelerated Gradient (NAG) algorithm to update the gradients. Default: False.
        filter_bias_and_bn: whether to filter batch norm parameters and bias from weight decay.
            If True, weight decay will not apply on BN parameters and bias in Conv or Dense layers. Default: True.
        loss_scale: A floating point value for the loss scale, which must be larger than 0.0. Default: 1.0.

    Returns:
        Optimizer object
    """
    opt = opt.lower()

    if weight_decay and filter_bias_and_bn:
        if not isinstance(params[0], dict):  # check whether param grouping strategy is encoded in `params`
            params = init_group_params(params, weight_decay)
        else:
            _logger.warning(
                "Customized param grouping strategy detected in `params`. "
                "filter_bias_and_bn (default=True) will be disabled"
            )

    # opt_args = dict(**kwargs)
    # if lr is not None:
    #    opt_args.setdefault('lr', lr)

    assert (
        loss_scale == 1.0
    ), "loss scale must be 1.0 in optimizer due to gradients are already scaled previously in TrainStepWrapper."

    # non-adaptive: SGD, momentum, and nesterov
    if opt == "sgd":
        # note: nn.Momentum may perform better if momentum > 0.
        opt_args = _collect_args(kwargs, nn.SGD)

        optimizer = nn.SGD(
            params=params,
            learning_rate=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            loss_scale=loss_scale,
            **opt_args,
        )
    elif opt in ["momentum", "nesterov"]:
        opt_args = _collect_args(kwargs, nn.Momentum)
        optimizer = nn.Momentum(
            params=params,
            learning_rate=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            use_nesterov=nesterov,
            loss_scale=loss_scale,
        )
    # adaptive
    elif opt == "adam":
        opt_args = _collect_args(kwargs, nn.Adam)
        optimizer = nn.Adam(
            params=params,
            learning_rate=lr,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
            use_nesterov=nesterov,
            **opt_args,
        )
    elif opt == "adamw":
        opt_args = _collect_args(kwargs, AdamW)
        optimizer = AdamW(
            params=params,
            learning_rate=lr,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
            **opt_args,
        )
    elif opt == "lion":
        opt_args = _collect_args(kwargs, Lion)
        optimizer = Lion(
            params=params,
            learning_rate=lr,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
            **opt_args,
        )
    elif opt == "nadam":
        opt_args = _collect_args(kwargs, NAdam)
        optimizer = NAdam(
            params=params,
            learning_rate=lr,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
            schedule_decay=schedule_decay,
            **opt_args,
        )
    elif opt == "adan":
        opt_args = _collect_args(kwargs, Adan)
        optimizer = Adan(
            params=params,
            learning_rate=lr,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
            **opt_args,
        )
    elif opt == "rmsprop":
        opt_args = _collect_args(kwargs, nn.RMSProp)
        optimizer = nn.RMSProp(
            params=params,
            learning_rate=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
            epsilon=eps,
            **opt_args,
        )
    elif opt == "adagrad":
        opt_args = _collect_args(kwargs, nn.Adagrad)
        optimizer = nn.Adagrad(
            params=params,
            learning_rate=lr,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
            **opt_args,
        )
    elif opt == "lamb":
        assert loss_scale == 1.0, "Loss scaler is not supported by Lamb optimizer"
        opt_args = _collect_args(kwargs, nn.Lamb)
        optimizer = nn.Lamb(
            params=params,
            learning_rate=lr,
            weight_decay=weight_decay,
            **opt_args,
        )
    else:
        raise ValueError(f"Invalid optimizer: {opt}")

    if os.path.exists(checkpoint_path):
        param_dict = load_checkpoint(checkpoint_path)
        load_param_into_net(optimizer, param_dict)

    return optimizer


def _collect_args(kwargs, optim_class):
    ret = {}
    valid_args = list(inspect.signature(optim_class.__init__).parameters.keys())[1:]  # remove self
    for arg in valid_args:
        assert arg != "clip", ValueError(
            "Gradient clipping should not be set in `optimizer`. Please set it in `train`."
        )
        if arg in kwargs:
            ret[arg] = kwargs[arg]
    return ret
