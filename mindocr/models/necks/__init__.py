from .fpn import FPN
from .select import Select

__all__ = ['build_neck']
support_necks = ['FPN', 'Select']


def build_neck(neck_name, **kwargs):
    assert neck_name in support_necks, f'Invalid neck: {neck_name}, Support necks are {support_necks}'
    neck = eval(neck_name)(**kwargs)
    return neck
