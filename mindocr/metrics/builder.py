from .det_metrics import *
from .rec_metrics import *
from . import det_metrics
from . import rec_metrics 

supported_metrics = det_metrics.__all__ + rec_metrics.__all__

# TODO: support multiple metrics
def build_metric(config):
    """
    Create the metric function.

    Args:
        config (dict): configuration for metric including metric `name` and also the kwargs specifically for each metric.
            - name (str): metric function name, exactly the same as one of the supported metric class names
    
    Return:
        nn.Metric
    
    Example: 
        >>> # Create a Metric for text recognition.
        >>> from mindocr.losses import build_loss
        >>> loss_func_name = "CTCLoss"
        >>> loss_func_config = {"pred_seq_len": 25, "max_label_len": 24, "batch_size": 32}
        >>> loss_fn = build_loss(loss_func_name, **loss_func_config)
        >>> loss_fn
        CTCLoss<>
    
    """

    mn = config.pop('name')
    if mn in supported_metrics:
        metric = eval(mn)(**config)
    else:
        raise ValueError(f'Invalid metric name {mn}, support metrics are {supported_metrics}')
    
    return metric
