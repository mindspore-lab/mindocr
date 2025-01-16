from .abinet_loss import ABINetLoss
from .cls_loss import CrossEntropySmooth
from .det_loss import DBLoss, EASTLoss, PSEDiceLoss
from .kie_loss import VQAReTokenLayoutLMLoss, VQASerTokenLayoutLMLoss
from .rec_loss import AttentionLoss, CTCLoss, SARLoss, VisionLANLoss
from .rec_multi_loss import MultiLoss
from .table_master_loss import TableMasterLoss
from .yolov8_loss import YOLOv8Loss

__all__ = ["build_loss"]

supported_losses = [
    "DBLoss",
    "CTCLoss",
    "AttentionLoss",
    "PSEDiceLoss",
    "EASTLoss",
    "CrossEntropySmooth",
    "ABINetLoss",
    "SARLoss",
    "VisionLANLoss",
    "VQAReTokenLayoutLMLoss",
    "VQASerTokenLayoutLMLoss",
    "YOLOv8Loss",
    "MultiLoss",
    "TableMasterLoss",
]


def build_loss(name, **kwargs):
    """
    Create the loss function.

    Args:
        name (str): loss function name, exactly the same as one of the supported loss class names

    Return:
        nn.LossBase

    Example:
        >>> # Create a CTC Loss module
        >>> from mindocr.losses import build_loss
        >>> loss_func_name = "CTCLoss"
        >>> loss_func_config = {"pred_seq_len": 25, "max_label_len": 24, "batch_size": 32}
        >>> loss_fn = build_loss(loss_func_name, **loss_func_config)
        >>> loss_fn
        CTCLoss<>
    """
    assert name in supported_losses, f"Invalid loss name {name}, support losses are {supported_losses}"

    loss_fn = eval(name)(**kwargs)

    # print('=> Loss func input args: \n\t', inspect.signature(loss_fn.construct))

    return loss_fn
