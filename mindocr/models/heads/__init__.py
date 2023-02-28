from .conv_head import ConvHead
from .det_db_head import DBHead
from .rec_ctc_head import CTCHead

__all__ = ['build_head']
support_heads = ['ConvHead', 'DBHead', 'CTCHead']


def build_head(head_name, **kwargs):
    assert head_name in support_heads, f'Invalid head {head_name}. Supported heads are {support_heads}'
    head = eval(head_name)(**kwargs)
    return head
