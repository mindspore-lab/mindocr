from .conv_head import ConvHead
from .det_db_head import DBHead
from .rec_ctc_head import CTCHead

__all__ = ['build_head']
supported_heads = ['ConvHead', 'DBHead', 'CTCHead']


def build_head(head_name, **kwargs):
    """
    Build Head network.
    
    Args:
        head_name (str): the head layer(s) name, which shoule be one of the supported_heads.
        kwargs (dict): input args for the head network
        
    Return:
        nn.Cell for head module
        
    Construct:
        Input: Tensor
        Output: Dict[Tensor]
        
    Example:
        >>> # build CTCHead
        >>> from mindocr.models.heads import build_head
        >>> config = dict(head_name='CTCHead', in_channels=256, out_channels=37)
        >>> head = build_head(**config)
        >>> print(head)
    """
    assert head_name in supported_heads, f'Invalid head {head_name}. Supported heads are {supported_heads}'
    head = eval(head_name)(**kwargs)
    return head
