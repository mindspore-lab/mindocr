from typing import List

from mindspore import Tensor, nn, ops

from ..utils.attention_cells import SEModule
from ._registry import register_backbone, register_backbone_class
from .mindcv_models.mobilenet_v3 import MobileNetV3, default_cfgs
from .mindcv_models.utils import load_pretrained

__all__ = ['DetMobileNetV3', 'det_mobilenet_v3']


@register_backbone_class
class DetMobileNetV3(MobileNetV3):
    """
    A wrapper of the original MobileNetV3 described in
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_ that extracts features maps from different
    stages.

    Args:
        arch: MobileNetV3 architecture type. Either 'small' or 'large'.
        out_stages: list of stage numbers from which to extract features.
        **kwargs: please check the parent class (MobileNetV3) for information.

    Examples:
        Initializing MobileNetV3 for feature extraction:
        >>> model = DetMobileNetV3("large", [5, 8, 14, 20])
    """
    def __init__(self, arch: str, *, out_stages: list = None, **kwargs):
        super().__init__(arch, **kwargs)
        del self.pool, self.classifier  # remove the original header to avoid confusion

        if out_stages is None:  # output the last stages if not specified
            out_stages = [len(self.features) - 1]
        self._out_stages = out_stages
        self.out_channels = [self._get_channels(stage) for stage in out_stages]

    def _get_channels(self, stage: int) -> int:
        """
        Find number of output channels at a stage.

        Args:
            stage: stage number.

        Returns:
            int: number of channels at the stage.
        """
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


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV3Enhance(nn.Cell):
    def __init__(self,
                 in_channels=3,
                 arch='large',
                 scale=0.5,
                 disable_se=False,
                 **kwargs):
        """
        the MobilenetV3 backbone network for detection module.
        Args:
            params(dict): the super parameters for build network
        """
        super().__init__()

        self.disable_se = disable_se

        if arch == "large":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', 2],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hardswish', 2],
                [3, 200, 80, False, 'hardswish', 1],
                [3, 184, 80, False, 'hardswish', 1],
                [3, 184, 80, False, 'hardswish', 1],
                [3, 480, 112, True, 'hardswish', 1],
                [3, 672, 112, True, 'hardswish', 1],
                [5, 672, 160, True, 'hardswish', 2],
                [5, 960, 160, True, 'hardswish', 1],
                [5, 960, 160, True, 'hardswish', 1],
            ]
            cls_ch_squeeze = 960
        elif arch == "small":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', 2],
                [3, 72, 24, False, 'relu', 2],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hardswish', 2],
                [5, 240, 40, True, 'hardswish', 1],
                [5, 240, 40, True, 'hardswish', 1],
                [5, 120, 48, True, 'hardswish', 1],
                [5, 144, 48, True, 'hardswish', 1],
                [5, 288, 96, True, 'hardswish', 2],
                [5, 576, 96, True, 'hardswish', 1],
                [5, 576, 96, True, 'hardswish', 1],
            ]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError("mode[" + arch +
                                      "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        if scale not in supported_scale:
            raise ValueError("supported scale are {} but input scale is {}".format(supported_scale, scale))
        inplanes = 16
        # conv1
        self.conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(inplanes * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            if_act=True,
            act='hardswish')

        self.stages = nn.CellList()
        self.out_channels = []
        block_list = []
        i = 0
        inplanes = make_divisible(inplanes * scale)
        for (k, exp, c, se, nl, s) in cfg:
            se = se and not self.disable_se
            start_idx = 2 if arch == 'large' else 0
            if s == 2 and i > start_idx:
                self.out_channels.append(inplanes)
                self.stages.append(nn.SequentialCell(*block_list))
                block_list = []
            block_list.append(
                ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=make_divisible(scale * exp),
                    out_channels=make_divisible(scale * c),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    act=nl))
            inplanes = make_divisible(scale * c)
            i += 1
        block_list.append(
            ConvBNLayer(
                in_channels=inplanes,
                out_channels=make_divisible(scale * cls_ch_squeeze),
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                if_act=True,
                act='hardswish'))
        self.stages.append(nn.SequentialCell(*block_list))
        self.out_channels.append(make_divisible(scale * cls_ch_squeeze))

    def construct(self, x):
        x = self.conv(x)
        out_list = []
        for stage in self.stages:
            x = stage(x)
            out_list.append(x)
        return out_list


class ConvBNLayer(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode='pad',
            padding=padding,
            group=groups,
            has_bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.hswish = nn.HSwish()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            if self.act == "relu":
                x = self.relu(x)
            elif self.act == "hardswish":
                x = self.hswish(x)
            else:
                print("The activation function({}) is selected incorrectly.".
                      format(self.act))
                exit()
        return x


class ResidualUnit(nn.Cell):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 use_se,
                 act=None):
        super(ResidualUnit, self).__init__()
        self.if_shortcut = stride == 1 and in_channels == out_channels
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act)
        self.bottleneck_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=int((kernel_size - 1) // 2),
            groups=mid_channels,
            if_act=True,
            act=act)
        if self.if_se:
            self.mid_se = SEModule(mid_channels)
        self.linear_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None)

    def construct(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = ops.add(inputs, x)
        return x


@register_backbone
def det_mobilenet_v3(architecture: str = "large", alpha: float = 1.0, in_channels: int = 3, pretrained: bool = True,
                     **kwargs) -> DetMobileNetV3:
    """
    A predefined MobileNetV3 for Text Detection.

    Args:
        architecture: MobileNetV3 architecture type. Either 'small' or 'large'.
        alpha: scale factor of model width. Default: 1.0.
        in_channels: number the channels of the input. Default: 3.
        pretrained: whether to load weights pretrained on ImageNet. Default: True.
        **kwargs: additional parameters to pass to MobileNetV3.

    Returns:
        DetMobileNetV3: MobileNetV3 model.
    """
    model = DetMobileNetV3(arch=architecture, alpha=alpha, in_channels=in_channels, **kwargs)

    if isinstance(pretrained, bool) and pretrained:
        name = f'mobilenet_v3_{architecture}_{alpha}'
        if name in default_cfgs:
            default_cfg = default_cfgs[name]
            load_pretrained(model, default_cfg)
        else:
            raise f'No pretrained {name} backbone found.'

    return model


@register_backbone
def det_mobilenet_v3_enhance(architecture: str = "large", alpha: float = 1.0, in_channels: int = 3,
                             pretrained: bool = True, **kwargs) -> DetMobileNetV3:
    """
    A predefined MobileNetV3 for Text Detection.

    Args:
        architecture: MobileNetV3 architecture type. Either 'small' or 'large'.
        alpha: scale factor of model width. Default: 1.0.
        in_channels: number the channels of the input. Default: 3.
        pretrained: whether to load weights pretrained on ImageNet. Default: True.
        **kwargs: additional parameters to pass to MobileNetV3.

    Returns:
        DetMobileNetV3: MobileNetV3 model.
    """
    model = MobileNetV3Enhance(arch=architecture, scale=alpha, in_channels=in_channels, **kwargs)

    if isinstance(pretrained, bool) and pretrained:
        name = f'mobilenet_v3_{architecture}_{alpha}'
        if name in default_cfgs:
            default_cfg = default_cfgs[name]
            load_pretrained(model, default_cfg)
        else:
            raise f'No pretrained {name} backbone found.'

    return model
