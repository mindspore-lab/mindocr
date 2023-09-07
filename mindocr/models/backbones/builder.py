from ..utils import load_model
from ._registry import backbone_class_entrypoint, backbone_entrypoint, is_backbone, is_backbone_class, list_backbones
from .mindcv_wrapper import MindCVBackboneWrapper

__all__ = ["build_backbone"]


def build_backbone(name, **kwargs):
    """
    Build the backbone network.

    Args:
        name (str): the backbone name, which can be a registered backbone class name
                        or a registered backbone (function) name.
        kwargs (dict): input args for the backbone
           1) if `name` is in the registered backbones (e.g. det_resnet50), kwargs include args for backbone creating
           likes `pretrained`
           2) if `name` is in the registered backbones class (e.g. DetResNet50), kwargs include args for the backbone
           configuration like `layers`.
           - pretrained: can be bool or str. If bool, load model weights from default url defined in the backbone py
           file. If str, pretrained can be url or local path to a checkpoint.


    Return:
        nn.Cell for backbone module

    Construct:
        Input: Tensor
        Output: List[Tensor]

    Example:
        >>> # build using backbone function name
        >>> from mindocr.models.backbones import build_backbone
        >>> backbone = build_backbone("det_resnet50", pretrained=True)
        >>> # build using backbone class name
        >>> from mindocr.models.backbones.mindcv_models.resnet import Bottleneck
        >>> cfg_from_class = dict(name="DetResNet", Bottleneck, layers=[3,4,6,3])
        >>> backbone = build_backbone(**cfg_from_class)
        >>> print(backbone)
    """
    remove_prefix = kwargs.pop("remove_prefix", False)


    if name in ["ClsMobileNetV3", "cls_mobilenet_v3_small_100"]:
        from .cls_mobilenet_v3 import ClsMobileNetV3, cls_mobilenet_v3_small_100
    if name in ["DetMobileNetV3", "det_mobilenet_v3"]:
        from .det_mobilenet import DetMobileNetV3, det_mobilenet_v3
    if name in ["DetResNet", "det_resnet50", "det_resnet18", "det_resnet152"]:
        from .det_resnet import DetResNet, det_resnet50, det_resnet18, det_resnet152
    if name in ["ABINetIterBackbone", "abinet_backbone"]:
        from .det_resnet import ABINetIterBackbone, abinet_backbone
    if name in ["rec_resnet_master_resnet31", "RecResNetMaster"]:
        from .rec_master import rec_resnet_master_resnet31, RecResNetMaster
    if name in ["RecResNet", "rec_resnet34", "rec_resnet31"]:
        from .rec_resnet import RecResNet, rec_resnet34, rec_resnet31
    if name in ["RecResNet45", "rec_resnet45"]:
        from .rec_resnet45 import RecResNet45, rec_resnet45
    if name in ["SVTRNet", "rec_svtr"]:
        from .rec_svtr import SVTRNet, rec_svtr
    if name in ["RecVGG", "rec_vgg7"]:
        from .rec_vgg import RecVGG, rec_vgg7

    if is_backbone(name):
        create_fn = backbone_entrypoint(name)
        backbone = create_fn(**kwargs)
    elif is_backbone_class(name):
        backbone_class = backbone_class_entrypoint(name)
        backbone = backbone_class(**kwargs)
    elif "mindcv" in name:
        # you can add `feature_only` parameter and `out_indices` in kwargs to extract intermediate features.
        backbone = MindCVBackboneWrapper(name, **kwargs)
    else:
        raise ValueError(f"Invalid backbone name: {name}, supported backbones are: {list_backbones()}")

    if "pretrained" in kwargs:
        pretrained = kwargs["pretrained"]
        if not isinstance(pretrained, bool):
            if remove_prefix:
                # remove the prefix with `backbone.`
                def fn(x): return {k.replace("backbone.", ""): v for k, v in x.items()}
            else:
                fn = None
            load_model(backbone, pretrained, filter_fn=fn)
        # No need to load again if pretrained is bool and True, because pretrained backbone is already loaded
        # in the backbone definition function.")

    return backbone
