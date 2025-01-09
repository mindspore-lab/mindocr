__all__ = ['build_head']
supported_heads = [
    'ConvHead',
    'DBHead',
    'DBHeadEnhance',
    'EASTHead',
    'CTCHead',
    'PSEHead',
    'AttentionHead',
    'MobileNetV3Head',
    'MasterDecoder',
    'RobustScannerHead',
    'VisionLANHead',
    'ABINetHead',
    "TokenClassificationHead",
    "RelationExtractionHead",
    'YOLOv8Head',
    'MultiHead',
    'TableMasterHead',
]
from .cls_head import MobileNetV3Head
from .conv_head import ConvHead
from .det_db_head import DBHead, DBHeadEnhance
from .det_east_head import EASTHead
from .det_pse_head import PSEHead
from .kie_relationextraction_head import RelationExtractionHead
from .kie_tokenclassification_head import TokenClassificationHead
from .rec_abinet_head import ABINetHead
from .rec_attn_head import AttentionHead
from .rec_ctc_head import CTCHead
from .rec_master_decoder import MasterDecoder
from .rec_multi_head import MultiHead
from .rec_robustscanner_head import RobustScannerHead
from .rec_visionlan_head import VisionLANHead
from .table_master_head import TableMasterHead
from .yolov8_head import YOLOv8Head


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
