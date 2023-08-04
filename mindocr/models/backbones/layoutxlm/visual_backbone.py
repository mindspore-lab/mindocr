import math
from dataclasses import dataclass
from typing import List, Optional, Type, Union

import numpy as np
import yaml

from mindspore import Tensor, nn, ops

from ..mindcv_models.resnet import BasicBlock, Bottleneck, ResNet, default_cfgs
from ..mindcv_models.utils import load_pretrained


def read_config():
    with open('visual_backbone.yaml', 'r') as file:
        data = yaml.safe_load(file)
    return data


@dataclass
class ShapeSpec:
    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None


class LayoutResNet(ResNet):
    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 num_classes: int,
                 out_features: List[str],
                 **kwargs):
        super().__init__(block, layers, num_classes, **kwargs)
        self._out_features = out_features
        curr_stride = self.conv1.stride
        self._out_feature_strides = {"stem": curr_stride}
        self._out_feature_channels = {"stem": self.conv1.out_channels}

        self.num_classes = num_classes
        self.stem = nn.SequentialCell([self.conv1, self.bn1, self.relu, self.max_pool])
        self.stage_names = ['res2', 'res3', 'res4', 'res5']
        self.stages = [self.layer1, self.layer2, self.layer3, self.layer4]
        for name, stage in zip(self.stage_names, self.stages):
            self._out_feature_strides[name] = curr_stride = int(
                curr_stride * np.prod([k.stride for k in stage])
            )
            self._out_feature_channels[name] = stage[-1].out_channels

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def construct(self, x: Tensor) -> dict:
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.pool(x)
            x = self.classifier(x)
            if 'linear' in self._out_features:
                outputs['linear'] = x

        return outputs


def layout_resnet101(pretrained: bool = True, **kwargs) -> LayoutResNet:
    """
    A predefined ResNet-101 for Text Detection.

    Args:
        pretrained: whether to load weights pretrained on ImageNet. Default: True.
        **kwargs: additional parameters to pass to ResNet.

    Returns:
        LayoutResNet: ResNet model.
    """
    model = LayoutResNet(BasicBlock, [3, 4, 23, 3], **kwargs)

    if pretrained:
        default_cfg = default_cfgs['resnet101']
        load_pretrained(model, default_cfg)

    return model


# ResNet101 - layoutxlm-base
def build_resnet_backbone(cfg):
    if cfg.MODEL.BACKBONE.NAME == "resnet101":
        return layout_resnet101(cfg.MODEL.BACKBONE.PRETRAINED)


def build_resnet_fpn_backbone(cfg):
    bottom_up = build_resnet_backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


class LastLevelMaxPool(nn.Cell):
    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def construct(self, x):
        return [ops.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class FPN(nn.Cell):
    def __init__(self,
                 bottom_up,
                 in_features,
                 out_channels,
                 norm="",
                 top_block=None,
                 fuse_type="sum",
                 square_pad=0):
        super(FPN, self).__init__()
        assert in_features, in_features

        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=use_bias)
            output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, has_bias=use_bias)
            stage = int(math.log2(strides[idx]))
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.top_block = top_block
        self.in_features = tuple(in_features)
        self.bottom_up = bottom_up

        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad
        self._fuse_type = fuse_type
        self._interpolate = ops.interpolate

    @property
    def size_divisibility(self):
        return self._size_divisibility

    @property
    def padding_constraints(self):
        return {"square_size": self._square_pad}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def construct(self, x):
        bottom_up_features = self.bottom_up(x)

        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        for idx, (lateral_conv, output_conv) in enumerate(zip(self.lateral_convs, self.output_convs)):
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                old_shape = list(prev_features.shape)
                new_size = tuple([2 * i for i in old_shape])
                top_down_features = self._interpolate(prev_features, size=new_size, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))

        assert len(self._out_features) == len(results)

        return {f: res for f, res in zip(self._out_features, results)}
