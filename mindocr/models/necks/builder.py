from .fpn import FPN, DBFPN, PSEFPN
from .rnn import RNNEncoder
from .select import Select

__all__ = ['build_neck']
supported_necks = ['FPN', 'DBFPN', 'PSEFPN', 'RNNEncoder', 'Select']


def build_neck(neck_name, **kwargs):
    """
    Build Neck network.
    
    Args:
        neck_name (str): the neck name, which shoule be one of the supported_necks.
        kwargs (dict): input args for the neck network
        
    Return:
        nn.Cell for neck module
        
    Construct:
        Input: Tensor
        Output: Dict[Tensor]
        
    Example:
        >>> # build RNNEncoder
        >>> from mindocr.models.necks import build_neck
        >>> config = dict(neck_name='RNNEncoder', in_channels=128, hidden_size=256)
        >>> neck = build_neck(**config)
        >>> print(neck)
    """
    assert neck_name in supported_necks, f'Invalid neck: {neck_name}, Support necks are {supported_necks}'
    neck = eval(neck_name)(**kwargs)
    return neck