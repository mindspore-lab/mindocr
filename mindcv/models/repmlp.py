"""
MindSpore implementation of `RepMLP`.
Refer to RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality.
"""

from collections import OrderedDict

import numpy as np

import mindspore.common.initializer as init
from mindspore import Tensor, nn, ops

from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "RepMLPNet",
    "RepMLPNet_T224",
    "RepMLPNet_T256",
    "RepMLPNet_B224",
    "RepMLPNet_B256",
    "RepMLPNet_D256",
    "RepMLPNet_L256",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "features.0",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "RepMLPNet_T224": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/repmlp/repmlp_t224-8dbedd00.ckpt"),
    "RepMLPNet_T256": _cfg(url="", input_size=(3, 256, 256)),
    "RepMLPNet_B224": _cfg(url=""),
    "RepMLPNet_B256": _cfg(url="", input_size=(3, 256, 256)),
    "RepMLPNet_D256": _cfg(url="", input_size=(3, 256, 256)),
    "RepMLPNet_L256": _cfg(url="", input_size=(3, 256, 256)),
}


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, group=1, has_bias=False):
    d = OrderedDict()
    conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      pad_mode="pad", padding=padding, group=group, has_bias=has_bias)
    bn1 = nn.BatchNorm2d(num_features=out_channels)
    d["conv"] = conv1
    d["bn"] = bn1
    result = nn.SequentialCell(d)
    return result


def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, group=1, has_bias=False):
    d = OrderedDict()
    conv2 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, group=group, has_bias=False)
    relu = nn.ReLU()
    d["conv"] = conv2
    d["relu"] = relu
    result = nn.SequentialCell(d)
    return result


def fuse_bn(conv_or_fc, bn):
    std = (bn.running_var + bn.eps).sqrt()
    t = bn.weight / std
    t = t.reshape(-1, 1, 1, 1)

    if len(t) == conv_or_fc.weight.size(0):
        return conv_or_fc.weight * t, bn.bias - bn.running_mean * bn.weight / std
    else:
        repeat_times = conv_or_fc.weight.size(0) // len(t)
        repeated = t.repeat_interleave(repeat_times, 0)
        return conv_or_fc.weight * repeated, (bn.bias - bn.running_mean * bn.weight / std).repeat_interleave(
            repeat_times, 0)


class GlobalPerceptron(nn.Cell):
    """GlobalPerceptron Layers provides global information(One of the three components of RepMLPBlock)"""

    def __init__(self, input_channels, internal_neurons):
        super(GlobalPerceptron, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=(1, 1), stride=1,
                             has_bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=(1, 1), stride=1,
                             has_bias=True)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.input_channels = input_channels
        self.shape = ops.Shape()

    def construct(self, x):
        shape = self.shape(x)
        pool = nn.AvgPool2d(kernel_size=(shape[2], shape[3]), stride=1)
        x = pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return x


class RepMLPBlock(nn.Cell):
    """Basic RepMLPBlock Layer(compose of Global Perceptron, Channel Perceptron and Local Perceptron)"""

    def __init__(self, in_channels, out_channels,
                 h, w,
                 reparam_conv_k=None,
                 globalperceptron_reduce=4,
                 num_sharesets=1,
                 deploy=False):
        super().__init__()

        self.C = in_channels  # noqa: E741
        self.O = out_channels  # noqa: E741
        self.S = num_sharesets  # noqa: E741

        self.h, self.w = h, w

        self.deploy = deploy
        self.transpose = ops.Transpose()
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()

        assert in_channels == out_channels
        self.gp = GlobalPerceptron(input_channels=in_channels, internal_neurons=in_channels // globalperceptron_reduce)

        self.fc3 = nn.Conv2d(in_channels=self.h * self.w * num_sharesets, out_channels=self.h * self.w * num_sharesets,
                             kernel_size=(1, 1), stride=1, padding=0, has_bias=deploy, group=num_sharesets)
        if deploy:
            self.fc3_bn = ops.Identity()
        else:
            self.fc3_bn = nn.BatchNorm2d(num_sharesets).set_train()

        self.reparam_conv_k = reparam_conv_k
        self.conv_branch_k = []
        if not deploy and reparam_conv_k is not None:
            for k in reparam_conv_k:
                conv_branch = conv_bn(num_sharesets, num_sharesets, kernel_size=k, stride=1, padding=k // 2,
                                      group=num_sharesets, has_bias=False)
                self.__setattr__("repconv{}".format(k), conv_branch)
                self.conv_branch_k.append(conv_branch)
                # print(conv_branch)

    def partition(self, x, h_parts, w_parts):
        x = x.reshape(-1, self.C, h_parts, self.h, w_parts, self.w)
        input_perm = (0, 2, 4, 1, 3, 5)
        x = self.transpose(x, input_perm)
        return x

    def partition_affine(self, x, h_parts, w_parts):
        fc_inputs = x.reshape(-1, self.S * self.h * self.w, 1, 1)
        out = self.fc3(fc_inputs)
        out = out.reshape(-1, self.S, self.h, self.w)
        out = self.fc3_bn(out)
        out = out.reshape(-1, h_parts, w_parts, self.S, self.h, self.w)
        return out

    def construct(self, inputs):
        # Global Perceptron
        global_vec = self.gp(inputs)

        origin_shape = self.shape(inputs)

        h_parts = origin_shape[2] // self.h
        w_parts = origin_shape[3] // self.w

        partitions = self.partition(inputs, h_parts, w_parts)

        #   Channel Perceptron
        fc3_out = self.partition_affine(partitions, h_parts, w_parts)

        #   Local Perceptron
        if self.reparam_conv_k is not None and not self.deploy:
            conv_inputs = self.reshape(partitions, (-1, self.S, self.h, self.w))
            conv_out = 0
            for k in self.conv_branch_k:
                conv_out += k(conv_inputs)
            conv_out = self.reshape(conv_out, (-1, h_parts, w_parts, self.S, self.h, self.w))
            fc3_out += conv_out

        input_perm = (0, 3, 1, 4, 2, 5)
        fc3_out = self.transpose(fc3_out, input_perm)  # N, O, h_parts, out_h, w_parts, out_w
        out = fc3_out.reshape(*origin_shape)
        out = out * global_vec
        return out

    def get_equivalent_fc3(self):
        fc_weight, fc_bias = fuse_bn(self.fc3, self.fc3_bn)
        if self.reparam_conv_k is not None:
            largest_k = max(self.reparam_conv_k)
            largest_branch = self.__getattr__("repconv{}".format(largest_k))
            total_kernel, total_bias = fuse_bn(largest_branch.conv, largest_branch.bn)
            for k in self.reparam_conv_k:
                if k != largest_k:
                    k_branch = self.__getattr__("repconv{}".format(k))
                    kernel, bias = fuse_bn(k_branch.conv, k_branch.bn)
                    total_kernel += nn.Pad(kernel, [(largest_k - k) // 2] * 4)
                    total_bias += bias
            rep_weight, rep_bias = self._convert_conv_to_fc(total_kernel, total_bias)
            final_fc3_weight = rep_weight.reshape_as(fc_weight) + fc_weight
            final_fc3_bias = rep_bias + fc_bias
        else:
            final_fc3_weight = fc_weight
            final_fc3_bias = fc_bias
        return final_fc3_weight, final_fc3_bias

    def local_inject(self):
        self.deploy = True
        #   Locality Injection
        fc3_weight, fc3_bias = self.get_equivalent_fc3()
        #   Remove Local Perceptron
        if self.reparam_conv_k is not None:
            for k in self.reparam_conv_k:
                self.__delattr__("repconv{}".format(k))
        self.__delattr__("fc3")
        self.__delattr__("fc3_bn")
        self.fc3 = nn.Conv2d(self.S * self.h * self.w, self.S * self.h * self.w, 1, 1, 0, has_bias=True, group=self.S)
        self.fc3_bn = ops.Identity()
        self.fc3.weight.data = fc3_weight
        self.fc3.bias.data = fc3_bias

    def _convert_conv_to_fc(self, conv_kernel, conv_bias):
        I = ops.eye(self.h * self.w).repeat(1, self.S).reshape(self.h * self.w, self.S, self.h, self.w)  # noqa: E741
        fc_k = ops.Conv2D(I, conv_kernel, pad=(conv_kernel.size(2) // 2, conv_kernel.size(3) // 2), group=self.S)
        fc_k = fc_k.reshape(self.h * self.w, self.S * self.h * self.w).t()
        fc_bias = conv_bias.repeat_interleave(self.h * self.w)
        return fc_k, fc_bias


class FFNBlock(nn.Cell):
    """Common FFN layer"""

    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_channels or in_channels
        hidden_features = hidden_channels or in_channels
        self.ffn_fc1 = conv_bn(in_channels, hidden_features, 1, 1, 0, has_bias=False)
        self.ffn_fc2 = conv_bn(hidden_features, out_features, 1, 1, 0, has_bias=False)
        self.act = act_layer()

    def construct(self, inputs):
        x = self.ffn_fc1(inputs)
        x = self.act(x)
        x = self.ffn_fc2(x)
        return x


class RepMLPNetUnit(nn.Cell):
    """Basic unit of RepMLPNet"""

    def __init__(self, channels, h, w, reparam_conv_k, globalperceptron_reduce, ffn_expand=4,
                 num_sharesets=1, deploy=False):
        super().__init__()
        self.repmlp_block = RepMLPBlock(in_channels=channels, out_channels=channels, h=h, w=w,
                                        reparam_conv_k=reparam_conv_k, globalperceptron_reduce=globalperceptron_reduce,
                                        num_sharesets=num_sharesets, deploy=deploy)
        self.ffn_block = FFNBlock(channels, channels * ffn_expand)
        self.prebn1 = nn.BatchNorm2d(channels).set_train()
        self.prebn2 = nn.BatchNorm2d(channels).set_train()

    def construct(self, x):
        y = x + self.repmlp_block(self.prebn1(x))
        # print(y)
        z = y + self.ffn_block(self.prebn2(y))
        return z


class RepMLPNet(nn.Cell):
    r"""RepMLPNet model class, based on
    `"RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality" <https://arxiv.org/pdf/2112.11081v2.pdf>`_

    Args:
        in_channels: number of input channels. Default: 3.
        num_classes: number of classification classes. Default: 1000.
        patch_size: size of a single image patch. Default: (4, 4)
        num_blocks: number of blocks per stage. Default: (2,2,6,2)
        channels: number of in_channels(channels[stage_idx]) and out_channels(channels[stage_idx + 1]) per stage.
            Default: (192,384,768,1536)
        hs: height of picture per stage. Default: (64,32,16,8)
        ws: width of picture per stage. Default: (64,32,16,8)
        sharesets_nums: number of share sets per stage. Default: (4,8,16,32)
        reparam_conv_k: convolution kernel size in local Perceptron. Default: (3,)
        globalperceptron_reduce: Intermediate convolution output size
            (in_channal = inchannal, out_channel = in_channel/globalperceptron_reduce) in globalperceptron. Default: 4
        use_checkpoint: whether to use checkpoint
        deploy: whether to use bias
    """

    def __init__(self,
                 in_channels=3, num_class=1000,
                 patch_size=(4, 4),
                 num_blocks=(2, 2, 6, 2), channels=(192, 384, 768, 1536),
                 hs=(64, 32, 16, 8), ws=(64, 32, 16, 8),
                 sharesets_nums=(4, 8, 16, 32),
                 reparam_conv_k=(3,),
                 globalperceptron_reduce=4, use_checkpoint=False,
                 deploy=False):
        super().__init__()
        num_stages = len(num_blocks)
        assert num_stages == len(channels)
        assert num_stages == len(hs)
        assert num_stages == len(ws)
        assert num_stages == len(sharesets_nums)

        self.conv_embedding = conv_bn_relu(in_channels, channels[0], kernel_size=patch_size, stride=patch_size,
                                           padding=0, has_bias=False)
        self.conv2d = nn.Conv2d(in_channels, channels[0], kernel_size=patch_size, stride=patch_size, padding=0)

        stages = []
        embeds = []
        for stage_idx in range(num_stages):
            stage_blocks = [RepMLPNetUnit(channels=channels[stage_idx], h=hs[stage_idx], w=ws[stage_idx],
                                          reparam_conv_k=reparam_conv_k,
                                          globalperceptron_reduce=globalperceptron_reduce, ffn_expand=4,
                                          num_sharesets=sharesets_nums[stage_idx],
                                          deploy=deploy) for _ in range(num_blocks[stage_idx])]
            stages.append(nn.CellList(stage_blocks))
            if stage_idx < num_stages - 1:
                embeds.append(
                    conv_bn_relu(in_channels=channels[stage_idx], out_channels=channels[stage_idx + 1], kernel_size=2,
                                 stride=2, padding=0))
        self.stages = nn.CellList(stages)
        self.embeds = nn.CellList(embeds)
        self.head_norm = nn.BatchNorm2d(channels[-1]).set_train()
        self.head = nn.Dense(channels[-1], num_class)

        self.use_checkpoint = use_checkpoint
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for cells."""
        for name, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                k = cell.group / (cell.in_channels * cell.kernel_size[0] * cell.kernel_size[1])
                k = k**0.5
                cell.weight.set_data(init.initializer(init.Uniform(k), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(init.Uniform(k), cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.Dense):
                k = 1 / cell.in_channels
                k = k**0.5
                cell.weight.set_data(init.initializer(init.Uniform(k), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(init.Uniform(k), cell.bias.shape, cell.bias.dtype))

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.conv_embedding(x)

        for i, stage in enumerate(self.stages):
            for block in stage:
                x = block(x)

            if i < len(self.stages) - 1:
                embed = self.embeds[i]
                x = embed(x)
        x = self.head_norm(x)
        shape = self.shape(x)
        pool = nn.AvgPool2d(kernel_size=(shape[2], shape[3]))
        x = pool(x)
        return x.view(shape[0], -1)

    def forward_head(self, x: Tensor) -> Tensor:
        return self.head(x)

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        return self.forward_head(x)


def locality_injection(self):
    for m in self.modules():
        if hasattr(m, "local_inject"):
            m.local_inject()


@register_model
def RepMLPNet_T224(pretrained: bool = False, image_size: int = 224, num_classes: int = 1000, in_channels=3,
                   deploy=False, **kwargs):
    """Get RepMLPNet_T224 model.
    Refer to the base class `models.RepMLPNet` for more details."""
    default_cfg = default_cfgs["RepMLPNet_T224"]
    model = RepMLPNet(in_channels=in_channels, num_class=num_classes, channels=(64, 128, 256, 512), hs=(56, 28, 14, 7),
                      ws=(56, 28, 14, 7),
                      num_blocks=(2, 2, 6, 2), reparam_conv_k=(1, 3), sharesets_nums=(1, 4, 16, 128),
                      deploy=deploy)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def RepMLPNet_T256(pretrained: bool = False, image_size: int = 256, num_classes: int = 1000, in_channels=3,
                   deploy=False, **kwargs):
    """Get RepMLPNet_T256 model.
    Refer to the base class `models.RepMLPNet` for more details."""
    default_cfg = default_cfgs["RepMLPNet_T256"]
    model = RepMLPNet(in_channels=in_channels, num_class=num_classes, channels=(64, 128, 256, 512), hs=(64, 32, 16, 8),
                      ws=(64, 32, 16, 8),
                      num_blocks=(2, 2, 6, 2), reparam_conv_k=(1, 3), sharesets_nums=(1, 4, 16, 128),
                      deploy=deploy)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def RepMLPNet_B224(pretrained: bool = False, image_size: int = 224, num_classes: int = 1000, in_channels=3,
                   deploy=False, **kwargs):
    """Get RepMLPNet_B224 model.
    Refer to the base class `models.RepMLPNet` for more details."""
    default_cfg = default_cfgs["RepMLPNet_B224"]
    model = RepMLPNet(in_channels=in_channels, num_class=num_classes, channels=(96, 192, 384, 768), hs=(56, 28, 14, 7),
                      ws=(56, 28, 14, 7),
                      num_blocks=(2, 2, 12, 2), reparam_conv_k=(1, 3), sharesets_nums=(1, 4, 32, 128),
                      deploy=deploy)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def RepMLPNet_B256(pretrained: bool = False, image_size: int = 256, num_classes: int = 1000, in_channels=3,
                   deploy=False, **kwargs):
    """Get RepMLPNet_B256 model.
    Refer to the base class `models.RepMLPNet` for more details."""
    default_cfg = default_cfgs["RepMLPNet_B256"]
    model = RepMLPNet(in_channels=in_channels, num_class=num_classes, channels=(96, 192, 384, 768), hs=(64, 32, 16, 8),
                      ws=(64, 32, 16, 8),
                      num_blocks=(2, 2, 12, 2), reparam_conv_k=(1, 3), sharesets_nums=(1, 4, 32, 128),
                      deploy=deploy)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def RepMLPNet_D256(pretrained: bool = False, image_size: int = 256, num_classes: int = 1000, in_channels=3,
                   deploy=False, **kwargs):
    """Get RepMLPNet_D256 model.
    Refer to the base class `models.RepMLPNet` for more details."""
    default_cfg = default_cfgs["RepMLPNet_D256"]
    model = RepMLPNet(in_channels=in_channels, num_class=num_classes, channels=(80, 160, 320, 640), hs=(64, 32, 16, 8),
                      ws=(64, 32, 16, 8),
                      num_blocks=(2, 2, 18, 2), reparam_conv_k=(1, 3), sharesets_nums=(1, 4, 16, 128),
                      deploy=deploy)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def RepMLPNet_L256(pretrained: bool = False, image_size: int = 256, num_classes: int = 1000, in_channels=3,
                   deploy=False, **kwargs):
    """Get RepMLPNet_L256 model.
    Refer to the base class `models.RepMLPNet` for more details."""
    default_cfg = default_cfgs["RepMLPNet_L256"]
    model = RepMLPNet(in_channels=in_channels, num_class=num_classes, channels=(96, 192, 384, 768), hs=(64, 32, 16, 8),
                      ws=(64, 32, 16, 8),
                      num_blocks=(2, 2, 18, 2), reparam_conv_k=(1, 3), sharesets_nums=(1, 4, 32, 256),
                      deploy=deploy)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


#   Verify the equivalency
if __name__ == "__main__":
    # x = Tensor(np.ones([1, 3, 3, 3]).astype(np.float32))
    dummy_input = Tensor(np.ones([1, 3, 256, 256]).astype(np.float32))
    model = RepMLPNet_B256()
    origin_y = model(dummy_input)

    model.locality_injection()
    print(model)
    print(origin_y)
