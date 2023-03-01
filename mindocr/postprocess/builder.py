from .det_postprocess import *
from .rec_postprocess import *
from . import det_postprocess
from . import rec_postprocess

supported_postprocess = det_postprocess.__all__ + rec_postprocess.__all__ 

def build_postprocess(config: dict):
    proc = config.pop('name')
    if proc in supported_postprocess:
        postprocessor = eval(proc)(**config)
    elif proc is None:
        return None
    else:
        raise ValueError(f'Invalid postprocess name {proc}, support postprocess are {supported_postprocess}')
    
    return postprocessor