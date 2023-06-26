__all__ = ["build_loss"]

supported_losses = [
    "L1BalancedCELoss",
    "CTCLoss",
    "AttentionLoss",
    "PSEDiceLoss",
    "EASTLoss",
    "CrossEntropySmooth",
    "FCELoss",
]
from .cls_loss import CrossEntropySmooth
from .det_loss import EASTLoss, FCELoss, L1BalancedCELoss, PSEDiceLoss
from .rec_loss import AttentionLoss, CTCLoss


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
