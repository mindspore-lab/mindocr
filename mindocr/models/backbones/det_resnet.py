from typing import List, Type, Union

from mindspore import Tensor

from ._registry import register_backbone, register_backbone_class
from .mindcv_models.resnet import BasicBlock, Bottleneck, ResNet, default_cfgs
from .mindcv_models.utils import load_pretrained

__all__ = ['DetResNet', 'det_resnet50', 'det_resnet18', 'det_resnet152']


@register_backbone_class
class DetResNet(ResNet):
    """
    A wrapper of the original ResNet described in
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_ that extracts features
    from stages 1 to 5 (4 features maps at different scales).

    Args:
        block: ResNet's building block.
        layers: number of layers in each stage.
        **kwargs: please check the parent class (ResNet) for information.

    Examples:
        Initializing ResNet-50 for feature extraction:
        >>> model = DetResNet(Bottleneck, [3, 4, 6, 3])
    """
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], **kwargs):
        super().__init__(block, layers, **kwargs)
        del self.pool, self.classifier  # remove the original header to avoid confusion
        self.out_channels = [ch * block.expansion for ch in [64, 128, 256, 512]]

    def construct(self, x: Tensor) -> List[Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        '''
        ftrs = []
        for i, layer in enumerate([self.layer1, self.layer2,  self.layer3, self.layer4]):
            x = layer(x)
            if i in self.out_indices:
                ftrs.append(x)
                self.out_channels.append()
        '''
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]


# TODO: load pretrained weight in build_backbone or use a unify wrapper to load


@register_backbone
def det_resnet18(pretrained: bool = True, **kwargs) -> DetResNet:
    """
    A predefined ResNet-18 for Text Detection.

    Args:
        pretrained: whether to load weights pretrained on ImageNet. Default: True.
        **kwargs: additional parameters to pass to ResNet.

    Returns:
        DetResNet: ResNet model.
    """
    model = DetResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained:
        default_cfg = default_cfgs['resnet18']
        load_pretrained(model, default_cfg)

    return model


@register_backbone
def det_resnet50(pretrained: bool = True, **kwargs) -> DetResNet:
    """
    A predefined ResNet-50 for Text Detection.

    Args:
        pretrained: whether to load weights pretrained on ImageNet. Default: True.
        **kwargs: additional parameters to pass to ResNet.

    Returns:
        DetResNet: ResNet model.
    """
    model = DetResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        default_cfg = default_cfgs['resnet50']
        load_pretrained(model, default_cfg)

    return model


@register_backbone
def det_resnet152(pretrained: bool = True, **kwargs) -> DetResNet:
    """
    A predefined ResNet-152 for Text Detection.

    Args:
        pretrained: whether to load weights pretrained on ImageNet. Default: True.
        **kwargs: additional parameters to pass to ResNet.

    Returns:
        DetResNet: ResNet model.
    """
    model = DetResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    if pretrained:
        default_cfg = default_cfgs['resnet152']
        load_pretrained(model, default_cfg)

    return model
