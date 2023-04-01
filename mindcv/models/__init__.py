"""models init"""
from . import (
    bit,
    cait,
    coat,
    convit,
    convnext,
    crossvit,
    densenet,
    dpn,
    edgenext,
    efficientnet,
    ghostnet,
    googlenet,
    hrnet,
    inception_v3,
    inception_v4,
    layers,
    mixnet,
    mlpmixer,
    mnasnet,
    mobilenet_v1,
    mobilenet_v2,
    mobilenet_v3,
    model_factory,
    nasnet,
    pit,
    pnasnet,
    poolformer,
    pvt,
    pvtv2,
    registry,
    regnet,
    repmlp,
    repvgg,
    res2net,
    resnest,
    resnet,
    resnetv2,
    rexnet,
    senet,
    shufflenetv1,
    shufflenetv2,
    sknet,
    squeezenet,
    swin_transformer,
    vgg,
    visformer,
    vit,
    xception,
    xcit,
)
from .bit import *
from .cait import *
from .coat import *
from .convit import *
from .convnext import *
from .crossvit import *
from .densenet import *
from .dpn import *
from .edgenext import *
from .efficientnet import *
from .ghostnet import *
from .googlenet import *
from .hrnet import *
from .inception_v3 import *
from .inception_v4 import *
from .layers import *
from .mixnet import *
from .mlpmixer import *
from .mnasnet import *
from .mobilenet_v1 import *
from .mobilenet_v2 import *
from .mobilenet_v3 import *
from .model_factory import *
from .nasnet import *
from .pit import *
from .pnasnet import *
from .poolformer import *
from .pvt import *
from .pvtv2 import *
from .registry import *
from .regnet import *
from .repmlp import *
from .repvgg import *
from .res2net import *
from .resnest import *
from .resnet import *
from .resnetv2 import *
from .rexnet import *
from .senet import *
from .shufflenetv1 import *
from .shufflenetv2 import *
from .sknet import *
from .squeezenet import *
from .swin_transformer import *
from .utils import *
from .vgg import *
from .visformer import *
from .vit import *
from .xception import *
from .xcit import *

# some net module is replaced by the net function with the same name when we do from .net import *
# we cannot use net.__all__, so we manually copy net.__all__ here.
__all__ = []
__all__.extend(bit.__all__)
__all__.extend(cait.__all__)
__all__.extend(coat.__all__)
__all__.extend(convit.__all__)
__all__.extend(convnext.__all__)
__all__.extend(crossvit.__all__)
__all__.extend(densenet.__all__)
__all__.extend(dpn.__all__)
__all__.extend(edgenext.__all__)
__all__.extend(efficientnet.__all__)
__all__.extend(ghostnet.__all__)
__all__.extend(["GoogLeNet", "googlenet"])
__all__.extend(hrnet.__all__)
__all__.extend(["InceptionV3", "inception_v3"])
__all__.extend(["InceptionV4", "inception_v4"])
__all__.extend(layers.__all__)
__all__.extend(mixnet.__all__)
__all__.extend(mlpmixer.__all__)
__all__.extend(mnasnet.__all__)
__all__.extend(mobilenet_v1.__all__)
__all__.extend(mobilenet_v2.__all__)
__all__.extend(mobilenet_v3.__all__)
__all__.extend(model_factory.__all__)
__all__.extend(nasnet.__all__)
__all__.extend(pit.__all__)
__all__.extend(["Pnasnet", "pnasnet"])
__all__.extend(poolformer.__all__)
__all__.extend(pvt.__all__)
__all__.extend(pvtv2.__all__)
__all__.extend(registry.__all__)
__all__.extend(regnet.__all__)
__all__.extend(repmlp.__all__)
__all__.extend(repvgg.__all__)
__all__.extend(res2net.__all__)
__all__.extend(resnest.__all__)
__all__.extend(resnet.__all__)
__all__.extend(resnetv2.__all__)
__all__.extend(rexnet.__all__)
__all__.extend(senet.__all__)
__all__.extend(shufflenetv1.__all__)
__all__.extend(shufflenetv2.__all__)
__all__.extend(sknet.__all__)
__all__.extend(squeezenet.__all__)
__all__.extend(swin_transformer.__all__)
__all__.extend(vgg.__all__)
__all__.extend(visformer.__all__)
__all__.extend(vit.__all__)
__all__.extend(["Xception", "xception"])
__all__.extend(xcit.__all__)
