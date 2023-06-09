from . import cls_metrics, det_metrics, rec_metrics
from .cls_metrics import *
from .det_metrics import *  # noqa
from .rec_metrics import *  # noqa

__all__ = ["build_metric"]

supported_metrics = det_metrics.__all__ + rec_metrics.__all__ + cls_metrics.__all__


def build_metric(config, device_num=1, **kwargs):
    """
    Create the metric function.

    Args:
        config (dict): configuration for metric including metric `name` and also the kwargs specifically for
            each metric.
            - name (str): metric function name, exactly the same as one of the supported metric class names
        device_num (int): number of devices. If device_num > 1, metric will be computed in distributed mode,
            i.e., aggregate intermediate variables (e.g., num_correct, TP) from all devices
            by `ops.AllReduce` op so as to correctly
            compute the metric on dispatched data.

    Return:
        nn.Metric

    Example:
        >>> # Create a RecMetric module for text recognition
        >>> from mindocr.metrics import build_metric
        >>> metric_config = {"name": "RecMetric", "main_indicator": "acc", "character_dict_path": None,
        "ignore_space": True, "print_flag": False}
        >>> metric = build_metric(metric_config)
        >>> metric
        <mindocr.metrics.rec_metrics.RecMetric>
    """

    mn = config.pop("name")
    if mn in supported_metrics:
        device_num = 1 if device_num is None else device_num
        config.update({"device_num": device_num})
        metric = eval(mn)(**config)
    else:
        raise ValueError(f"Invalid metric name {mn}, support metrics are {supported_metrics}")

    return metric
