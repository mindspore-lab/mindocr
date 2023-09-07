__all__ = ['build_head']

import logging


supported_heads = [
    'ConvHead',
    'DBHead',
    'EASTHead',
    'CTCHead',
    'PSEHead',
    'AttentionHead',
    'MobileNetV3Head',
    'FCEHead',
    'MasterDecoder',
    'RobustScannerHead',
    'VisionLANHead',
    'ABINetHead',
]

_logger = logging.getLogger(__name__)

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
    if head_name not in supported_heads:
        _logger.error(f'Invalid head {head_name}. Supported heads are {supported_heads}')
        exit(1)
    if head_name == 'ConvHead':
        from .conv_head import ConvHead
    if head_name == 'DBHead':
        from .det_db_head import DBHead
    if head_name == 'EASTHead':
        from .det_east_head import EASTHead
    if head_name == 'CTCHead':
        from .rec_ctc_head import CTCHead
    if head_name == 'PSEHead':
        from .det_pse_head import PSEHead
    if head_name == 'AttentionHead':
        from .rec_attn_head import AttentionHead
    if head_name == 'MobileNetV3Head':
        from .cls_head import MobileNetV3Head
    if head_name == 'FCEHead':
        from .det_fce_head import FCEHead
    if head_name == 'MasterDecoder':
        from .rec_master_decoder import MasterDecoder
    if head_name == 'RobustScannerHead':
        from .rec_robustscanner_head import RobustScannerHead
    if head_name == 'VisionLANHead':
        from .rec_visionlan_head import VisionLANHead
    if head_name == 'ABINetHead':
        from .rec_abinet_head import ABINetHead
    head = eval(head_name)(**kwargs)
    return head
