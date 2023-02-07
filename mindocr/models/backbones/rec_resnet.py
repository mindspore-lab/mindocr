from typing import Tuple, List
from mindspore import Tensor
from mindcv.models.resnet import ResNet, BasicBlock, load_pretrained, default_cfgs
from ._registry import register_backbone, register_backbone_class

__all__ = ['RecResNet', 'rec_resnet34']


@register_backbone_class
class RecResNet(ResNet):
    def __init__(self, block, layers, **kwargs):
        super().__init__(block, layers, **kwargs)
        del self.pool, self.classifier  # remove the original header to avoid confusion
        self.out_channels = 512

    def construct(self, x: Tensor) -> List[Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

# TODO: load pretrained weight in build_backbone or use a unify wrapper to load


@register_backbone
def rec_resnet34(pretrained: bool = True, **kwargs):
    model = RecResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['resnet34']
        load_pretrained(model, default_cfg)

    return model
