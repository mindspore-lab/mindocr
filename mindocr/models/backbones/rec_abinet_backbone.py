import math

import numpy as np

import mindspore as ms
import mindspore.nn as nn

from ..utils.abinet_layers import (
    ABINetBlock,
    PositionalEncoding,
    PositionAttention,
    TransformerEncoder,
    _default_tfmer_cfg,
)
from ._registry import register_backbone, register_backbone_class

__all__ = [
    "ABINetIterBackbone",
    "abinet_backbone"]

# ABINet_backbone


@register_backbone_class
class ABINetIterBackbone(nn.Cell):
    def __init__(self, batchsize=96):
        super().__init__()
        self.out_channels = [1, 512]
        self.batchsize = batchsize
        self.vision = BaseVision(self.batchsize)

    def construct(self, images, *args):
        v_res = self.vision(images)
        return v_res


@register_backbone
def abinet_backbone(pretrained: bool = True, **kwargs):
    model = ABINetIterBackbone(**kwargs)

    # load pretrained weights
    if pretrained:
        raise NotImplementedError("The default pretrained checkpoint for `rec_abinet_backbone` backbone does not exist")

    return model


class BaseVision(ABINetBlock):
    def __init__(self, batchsize):
        super().__init__()
        self.batchsize = batchsize
        self.loss_weight = 1.0
        self.out_channels = 512
        self.backbone = ResTranformer(self.batchsize)
        mode = "nearest"
        self.attention = PositionAttention(
            max_length=26,  # additional stop token
            mode=mode,
        )

        self.cls = nn.Dense(
            self.out_channels,
            self.charset.num_classes,
            weight_init="HeUniform",
            bias_init="uniform",
        )

    def construct(self, images, *args):
        features = self.backbone(images)  # (N, E, H, W)

        attn_vecs, attn_scores = self.attention(features)

        logits = self.cls(attn_vecs)  # (N, T, C)

        pt_lengths = self._get_length(logits)

        return {
            "feature": attn_vecs,
            "logits": logits,
            "pt_lengths": pt_lengths,
            "attn_scores": attn_scores,
            "loss_weight": self.loss_weight,
            "name": "vision",
        }


class ResTranformer(nn.Cell):
    def __init__(self, batchsize):
        super().__init__()
        self.resnet = resnet45()

        self.d_model = _default_tfmer_cfg["d_model"]
        nhead = _default_tfmer_cfg["nhead"]
        d_inner = _default_tfmer_cfg["d_inner"]
        dropout = _default_tfmer_cfg["dropout"]
        num_layers = 3
        self.encoder_mask = ms.Tensor(np.ones((batchsize, 256, 256)), dtype=ms.float32)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=8 * 32)

        self.transformer = TransformerEncoder(
            batch_size=batchsize,
            num_layers=num_layers,
            hidden_size=self.d_model,
            num_heads=nhead,
            ffn_hidden_size=d_inner,
            hidden_dropout_rate=dropout,
            attention_dropout_rate=dropout,
            hidden_act="relu",
            seq_length=256,
        )

    def construct(self, images):
        feature = self.resnet(images)
        n, c, h, w = feature.shape
        feature = feature.view(n, c, -1)
        feature = feature.transpose(2, 0, 1)

        feature = self.pos_encoder(feature)
        feature = feature.transpose(1, 0, 2)
        feature = self.transformer(
            feature, self.encoder_mask
        )
        feature = feature.transpose(1, 0, 2)
        feature = feature.transpose(1, 2, 0)
        feature = feature.view(n, c, h, w)
        return feature


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, has_bias=False
    )


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        pad_mode="pad",
        padding=1,
        has_bias=False,
    )


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    def __init__(self, block, layers):
        self.inplanes = 32
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=3, stride=1, padding=1, has_bias=False, pad_mode="pad"
        )

        self.bn1 = nn.BatchNorm2d(32, momentum=0.1)
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(block, 32, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1)
        self.layer5 = self._make_layer(block, 512, layers[4], stride=1)

        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                n = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                cell.weight.set_data(
                    ms.common.initializer.initializer(
                        ms.common.initializer.Normal(sigma=math.sqrt(2.0 / n), mean=0),
                        cell.weight.shape,
                        cell.weight.dtype,
                    )
                )
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(
                    ms.common.initializer.initializer(
                        "ones", cell.gamma.shape, cell.gamma.dtype
                    )
                )
                cell.beta.set_data(
                    ms.common.initializer.initializer(
                        "zeros", cell.beta.shape, cell.beta.dtype
                    )
                )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    has_bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


def resnet45():
    return ResNet(BasicBlock, [3, 4, 6, 6, 3])
