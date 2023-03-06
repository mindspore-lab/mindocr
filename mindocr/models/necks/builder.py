from .fpn import FPN, DBFPN
from .rnn import RNNEncoder
from .select import Select

__all__ = ['build_neck']
supported_necks = ['FPN', 'DBFPN', 'RNNEncoder', 'Select']


def build_neck(neck_name, **kwargs):
    assert neck_name in supported_necks, f'Invalid neck: {neck_name}, Support necks are {supported_necks}'
    neck = eval(neck_name)(**kwargs)
    return neck