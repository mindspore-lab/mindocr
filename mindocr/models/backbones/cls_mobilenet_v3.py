from mindspore import Tensor

from ._registry import register_backbone, register_backbone_class
from .mindcv_models.mobilenet_v3 import MobileNetV3, default_cfgs
from .mindcv_models.utils import load_pretrained

__all__ = ['ClsMobileNetV3', 'cls_mobilenet_v3_small_100']


@register_backbone_class
class ClsMobileNetV3(MobileNetV3):
    def __init__(
            self,
            arch: str,
            alpha: float = 1.0,
            round_nearest: int = 8,
            in_channels: int = 3) -> None:
        super().__init__(arch, alpha, round_nearest, in_channels)
        self.out_channels = [self.feature_info[-1]['chs']]

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        return x


@register_backbone
def cls_mobilenet_v3_small_100(pretrained: bool = True, in_channels: int = 3, **kwargs):
    """Get small MobileNetV3 model without width scaling.
    """
    model = ClsMobileNetV3(arch="small", alpha=1.0, in_channels=in_channels, **kwargs)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['mobilenet_v3_small_1.0']
        load_pretrained(model, default_cfg)

    return model
