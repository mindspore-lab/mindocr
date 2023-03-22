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
        >>> # Create a RecMetric module for text recognition
        >>> from mindocr.metrics import build_metric
        >>> metric_config = {"name": "RecMetric", "main_indicator": "acc", "character_dict_path": None, "ignore_space": True, "print_flag": False}
        >>> metric = build_metric(metric_config)
        >>> metric
        <mindocr.metrics.rec_metrics.RecMetric>
    """

    mn = config.pop('name')
    if mn in supported_metrics:
        metric = eval(mn)(**config)
    else:
        raise ValueError(f'Invalid metric name {mn}, support metrics are {supported_metrics}')
    
    return metric
