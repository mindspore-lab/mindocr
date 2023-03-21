import inspect
from .det_loss import L1BalancedCELoss
from .rec_loss import CTCLoss

supported_losses = ['L1BalancedCELoss', 'CTCLoss']


def build_loss(name, **kwargs):
    """
    Create the loss function.

    Args:
        name (str): loss function name, exactly the same as one of the supported loss class names
    
    Return:
        nn.LossBase
    
    Example:
        >>> # Create a CTC Loss module
        >>> from mindocr.metrics import build_metric
        >>> metric_name = "RecMetric"
        >>> metric_config = {"name": metric_name, "main_indicator": "acc", "character_dict_path": None, "ignore_space": True, "print_flag": False}
        >>> metric = build_metric(metric_config)
        >>> metric
        <mindocr.metrics.rec_metrics.RecMetric>
    """
    assert name in supported_losses, f'Invalid loss name {name}, support losses are {supported_losses}'

    loss_fn = eval(name)(**kwargs)

    # print('loss func inputs: ', loss_fn.construct.__code__.co_varnames)
    print('==> Loss func input args: \n\t', inspect.signature(loss_fn.construct))

    return loss_fn
