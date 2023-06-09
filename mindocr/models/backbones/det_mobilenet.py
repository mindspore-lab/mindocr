from typing import List

from mindspore import Tensor

from ._registry import register_backbone, register_backbone_class
from .mindcv_models.mobilenet_v3 import MobileNetV3, default_cfgs
from .mindcv_models.utils import load_pretrained

__all__ = ['DetMobileNetV3', 'det_mobilenet_v3']


@register_backbone_class
class DetMobileNetV3(MobileNetV3):
    def __init__(self, out_stages: list = None, **kwargs):
        super().__init__(**kwargs)
        del self.pool, self.classifier  # remove the original header to avoid confusion

        if out_stages is None:  # output the last stages if not specified
            out_stages = [len(self.features) - 1]
        self._out_stages = out_stages
        self.out_channels = [self._get_channels(stage) for stage in out_stages]

    def _get_channels(self, stage):
        params = list(self.features[stage].get_parameters())
        if not params:
            return self._get_channels(stage - 1)
        return params[-1].shape[0]

    def construct(self, x: Tensor) -> List[Tensor]:
        output = []
        for i, feature in enumerate(self.features):
            x = feature(x)
            if i in self._out_stages:
                output.append(x)

        return output


@register_backbone
def det_mobilenet_v3(architecture='large', alpha=1.0, in_channels=3, pretrained: bool = True, **kwargs):
    model = DetMobileNetV3(arch=architecture, alpha=alpha, in_channels=in_channels, **kwargs)

    # load pretrained weights
    if isinstance(pretrained, bool) and pretrained:
        name = f'mobilenet_v3_{architecture}_{alpha}'
        if name in default_cfgs:
            default_cfg = default_cfgs[name]
            load_pretrained(model, default_cfg)
        else:
            raise f'No pretrained {name} backbone found.'

    return model
