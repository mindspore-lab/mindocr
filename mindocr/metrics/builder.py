from .det_metrics import *
from .rec_metrics import *
from . import det_metrics
from . import rec_metrics 

supported_metrics = det_metrics.__all__ + rec_metrics.__all__

# TODO: support multiple metrics
def build_metric(config):
    mn = config.pop('name')
    if mn in supported_metrics:
        metric = eval(mn)(**config)
    else:
        raise ValueError(f'Invalid metric name {mn}, support metrics are {supported_metrics}')
    
    return metric
