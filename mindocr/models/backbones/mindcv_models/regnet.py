"""
MindSpore implementation of `RegNet`.
Refer to: Designing Network Design Spaces
"""

import math

import numpy as np

import mindspore.common.initializer as init
from mindspore import nn

from .layers.pooling import GlobalAvgPooling
from .layers.squeeze_excite import SqueezeExcite
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "regnet_x_200mf",
    "regnet_x_400mf",
    "regnet_x_600mf",
    "regnet_x_800mf",
    "regnet_x_1_6gf",
    "regnet_x_3_2gf",
    "regnet_x_4_0gf",
    "regnet_x_6_4gf",
    "regnet_x_8_0gf",
    "regnet_x_12gf",
    "regnet_x_16gf",
    "regnet_x_32gf",
    "regnet_y_200mf",
    "regnet_y_400mf",
    "regnet_y_600mf",
    "regnet_y_800mf",
    "regnet_y_1_6gf",
    "regnet_y_3_2gf",
    "regnet_y_4_0gf",
    "regnet_y_6_4gf",
    "regnet_y_8_0gf",
    "regnet_y_12gf",
    "regnet_y_16gf",
    "regnet_y_32gf",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "regnet_x_200mf": _cfg(url=""),
    "regnet_x_400mf": _cfg(url=""),
    "regnet_x_600mf": _cfg(url=""),
    "regnet_x_800mf": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/regnet/regnet_x_800mf-617227f4.ckpt"),
    "regnet_x_1_6gf": _cfg(url=""),
    "regnet_x_3_2gf": _cfg(url=""),
    "regnet_x_4_0gf": _cfg(url=""),
    "regnet_x_6_4gf": _cfg(url=""),
    "regnet_x_8_0gf": _cfg(url=""),
    "regnet_x_12gf": _cfg(url=""),
    "regnet_x_16gf": _cfg(url=""),
    "regnet_x_32gf": _cfg(url=""),
    "regnet_y_200mf": _cfg(url=""),
    "regnet_y_400mf": _cfg(url=""),
    "regnet_y_600mf": _cfg(url=""),
    "regnet_y_800mf": _cfg(url=""),
    "regnet_y_1_6gf": _cfg(url=""),
    "regnet_y_3_2gf": _cfg(url=""),
    "regnet_y_4_0gf": _cfg(url=""),
    "regnet_y_6_4gf": _cfg(url=""),
    "regnet_y_8_0gf": _cfg(url=""),
    "regnet_y_12gf": _cfg(url=""),
    "regnet_y_16gf": _cfg(url=""),
    "regnet_y_32gf": _cfg(url=""),
}


def conv2d(w_in, w_out, k, *, stride=1, groups=1, bias=False):
    """Helper for building a conv2d layer."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    s, p, g, b = stride, (k - 1) // 2, groups, bias
    return nn.Conv2d(w_in, w_out, k, stride=s, pad_mode="pad", padding=p, group=g, has_bias=b)


def norm2d(w_in, eps=1e-5, mom=0.9):
    """Helper for building a norm2d layer."""
    return nn.BatchNorm2d(num_features=w_in, eps=eps, momentum=mom)


def pool2d(_w_in, k, *, stride=1):
    """Helper for building a pool2d layer."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    padding = (k - 1) // 2
    pad2d = nn.Pad(((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="CONSTANT")
    max_pool = nn.MaxPool2d(kernel_size=k, stride=stride, pad_mode="valid")
    return nn.SequentialCell([pad2d, max_pool])


def gap2d(keep_dims=False):
    """Helper for building a gap2d layer."""
    return GlobalAvgPooling(keep_dims)


def linear(w_in, w_out, *, bias=False):
    """Helper for building a linear layer."""
    return nn.Dense(w_in, w_out, has_bias=bias)


def activation():
    """Helper for building an activation layer."""
    return nn.ReLU()


class ResStemCifar(nn.Cell):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(ResStemCifar, self).__init__()
        self.conv = conv2d(w_in, w_out, 3)
        self.bn = norm2d(w_out)
        self.af = activation()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        return x


class ResStem(nn.Cell):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStem, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.pool = pool2d(w_out, 3, stride=2)

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.pool(x)
        return x


class SimpleStem(nn.Cell):
    """Simple stem for ImageNet: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(SimpleStem, self).__init__()
        self.conv = conv2d(w_in, w_out, 3, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        return x


class VanillaBlock(nn.Cell):
    """Vanilla block: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride, _params):
        super(VanillaBlock, self).__init__()
        self.a = conv2d(w_in, w_out, 3, stride=stride)
        self.a_bn = norm2d(w_out)
        self.a_af = activation()
        self.b = conv2d(w_out, w_out, 3)
        self.b_bn = norm2d(w_out)
        self.b_af = activation()

    def construct(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_af(x)
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_af(x)
        return x


class BasicTransform(nn.Cell):
    """Basic transformation: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride, _params):
        super(BasicTransform, self).__init__()
        self.a = conv2d(w_in, w_out, 3, stride=stride)
        self.a_bn = norm2d(w_out)
        self.a_af = activation()
        self.b = conv2d(w_out, w_out, 3)
        self.b_bn = norm2d(w_out)
        self.b_bn.final_bn = True

    def construct(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_af(x)
        x = self.b(x)
        x = self.b_bn(x)
        return x


class ResBasicBlock(nn.Cell):
    """Residual basic block: x + f(x), f = basic transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(ResBasicBlock, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out)
        self.f = BasicTransform(w_in, w_out, stride, params)
        self.af = activation()

    def construct(self, x):
        x_p = self.bn(self.proj(x)) if self.proj is not None else x
        return self.af(x_p + self.f(x))


class BottleneckTransform(nn.Cell):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, w_in, w_out, stride, params):
        super(BottleneckTransform, self).__init__()
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        self.a = conv2d(w_in, w_b, 1)
        self.a_bn = norm2d(w_b)
        self.a_af = activation()
        self.b = conv2d(w_b, w_b, 3, stride=stride, groups=groups)
        self.b_bn = norm2d(w_b)
        self.b_af = activation()
        self.se = SqueezeExcite(in_channels=w_b, rd_channels=w_se) if w_se else None
        self.c = conv2d(w_b, w_out, 1)
        self.c_bn = norm2d(w_out)
        self.c_bn.final_bn = True

    def construct(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_af(x)
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_af(x)
        x = self.se(x) if self.se is not None else x
        x = self.c(x)
        x = self.c_bn(x)
        return x


class ResBottleneckBlock(nn.Cell):
    """Residual bottleneck block: x + f(x), f = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(ResBottleneckBlock, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out)
        self.f = BottleneckTransform(w_in, w_out, stride, params)
        self.af = activation()

    def construct(self, x):
        x_p = self.bn(self.proj(x)) if self.proj is not None else x
        return self.af(x_p + self.f(x))


class ResBottleneckLinearBlock(nn.Cell):
    """Residual linear bottleneck block: x + f(x), f = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(ResBottleneckLinearBlock, self).__init__()
        self.has_skip = (w_in == w_out) and (stride == 1)
        self.f = BottleneckTransform(w_in, w_out, stride, params)

    def construct(self, x):
        return x + self.f(x) if self.has_skip else self.f(x)


class AnyStage(nn.Cell):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, params):
        super(AnyStage, self).__init__()
        self.blocks = nn.CellList()
        for _ in range(d):
            block = block_fun(w_in, w_out, stride, params)
            self.blocks.append(block)
            stride, w_in = 1, w_out

    def construct(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class AnyHead(nn.Cell):
    """AnyNet head: optional conv, AvgPool, 1x1."""

    def __init__(self, w_in, head_width, num_classes):
        super(AnyHead, self).__init__()
        self.head_width = head_width
        if head_width > 0:
            self.conv = conv2d(w_in, head_width, 1)
            self.bn = norm2d(head_width)
            self.af = activation()
            w_in = head_width
        self.avg_pool = gap2d()
        self.fc = linear(w_in, num_classes, bias=True)

    def construct(self, x):
        x = self.af(self.bn(self.conv(x))) if self.head_width > 0 else x
        x = self.avg_pool(x)
        x = self.fc(x)
        return x


def get_stem_fun(stem_type):
    """Retrieves the stem function by name."""
    stem_funs = {
        "res_stem_cifar": ResStemCifar,
        "res_stem_in": ResStem,
        "simple_stem_in": SimpleStem,
    }
    err_str = "Stem type '{}' not supported"
    assert stem_type in stem_funs.keys(), err_str.format(stem_type)
    return stem_funs[stem_type]


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {
        "vanilla_block": VanillaBlock,
        "res_basic_block": ResBasicBlock,
        "res_bottleneck_block": ResBottleneckBlock,
        "res_bottleneck_linear_block": ResBottleneckLinearBlock,
    }
    err_str = "Block type '{}' not supported"
    assert block_type in block_funs.keys(), err_str.format(block_type)
    return block_funs[block_type]


class AnyNet(nn.Cell):
    """AnyNet model."""

    @staticmethod
    def anynet_get_params(depths, stem_type, stem_w, block_type, widths, strides, bot_muls, group_ws, head_w,
                          num_classes, se_r):
        nones = [None for _ in depths]
        return {
            "stem_type": stem_type,
            "stem_w": stem_w,
            "block_type": block_type,
            "depths": depths,
            "widths": widths,
            "strides": strides,
            "bot_muls": bot_muls if bot_muls else nones,
            "group_ws": group_ws if group_ws else nones,
            "head_w": head_w,
            "se_r": se_r,
            "num_classes": num_classes,
        }

    def __init__(self, depths, stem_type, stem_w, block_type, widths, strides, bot_muls, group_ws, head_w, num_classes,
                 se_r, in_channels):
        super(AnyNet, self).__init__()
        p = AnyNet.anynet_get_params(depths, stem_type, stem_w, block_type, widths, strides, bot_muls, group_ws, head_w,
                                     num_classes, se_r)
        stem_fun = get_stem_fun(p["stem_type"])
        block_fun = get_block_fun(p["block_type"])
        self.stem = stem_fun(in_channels, p["stem_w"])
        prev_w = p["stem_w"]
        keys = ["depths", "widths", "strides", "bot_muls", "group_ws"]
        self.stages = nn.CellList()
        for i, (d, w, s, b, g) in enumerate(zip(*[p[k] for k in keys])):
            params = {"bot_mul": b, "group_w": g, "se_r": p["se_r"]}
            stage = AnyStage(prev_w, w, s, d, block_fun, params)
            self.stages.append(stage)
            prev_w = w
        self.head = AnyHead(prev_w, p["head_w"], p["num_classes"])
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                fan_out = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                cell.weight.set_data(
                    init.initializer(init.Normal(sigma=math.sqrt(2.0 / fan_out), mean=0.0),
                                     cell.weight.shape, cell.weight.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(init.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(init.initializer("zeros", cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(init.Normal(sigma=0.01, mean=0.0), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def forward_features(self, x):
        x = self.stem(x)
        for module in self.stages:
            x = module(x)
        return x

    def forward_head(self, x):
        x = self.head(x)
        return x

    def construct(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def adjust_block_compatibility(ws, bs, gs):
    """Adjusts the compatibility of widths, bottlenecks, and groups."""
    assert len(ws) == len(bs) == len(gs)
    assert all(w > 0 and b > 0 and g > 0 for w, b, g in zip(ws, bs, gs))
    assert all(b < 1 or b % 1 == 0 for b in bs)
    vs = [int(max(1, w * b)) for w, b in zip(ws, bs)]
    gs = [int(min(g, v)) for g, v in zip(gs, vs)]
    ms = [np.lcm(g, int(b)) if b > 1 else g for g, b in zip(gs, bs)]
    vs = [max(m, int(round(v / m) * m)) for v, m in zip(vs, ms)]
    ws = [int(v / b) for v, b in zip(vs, bs)]
    assert all(w * b % g == 0 for w, b, g in zip(ws, bs, gs))
    return ws, bs, gs


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per stage widths and depths from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    # Generate continuous per-block ws
    ws_cont = np.arange(d) * w_a + w_0
    # Generate quantized per-block ws
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws_all = w_0 * np.power(w_m, ks)
    ws_all = np.round(np.divide(ws_all, q)).astype(int) * q
    # Generate per stage ws and ds (assumes ws_all are sorted)
    ws, ds = np.unique(ws_all, return_counts=True)
    # Compute number of actual stages and total possible stages
    num_stages, total_stages = len(ws), ks.max() + 1
    # Convert numpy arrays to lists and return
    ws, ds, ws_all, ws_cont = (x.tolist() for x in (ws, ds, ws_all, ws_cont))
    return ws, ds, num_stages, total_stages, ws_all, ws_cont


def generate_regnet_full(w_a, w_0, w_m, d, stride, bot_mul, group_w):
    """Generates per stage ws, ds, gs, bs, and ss from RegNet cfg."""
    ws, ds = generate_regnet(w_a, w_0, w_m, d)[0:2]
    ss = [stride for _ in ws]
    bs = [bot_mul for _ in ws]
    gs = [group_w for _ in ws]
    ws, bs, gs = adjust_block_compatibility(ws, bs, gs)
    return ws, ds, ss, bs, gs


class RegNet(AnyNet):
    r"""RegNet model class, based on
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_
    """

    @staticmethod
    def regnet_get_params(w_a, w_0, w_m, d, stride, bot_mul, group_w, stem_type, stem_w, block_type, head_w,
                          num_classes, se_r):
        """Get AnyNet parameters that correspond to the RegNet."""
        ws, ds, ss, bs, gs = generate_regnet_full(w_a, w_0, w_m, d, stride, bot_mul, group_w)
        return {
            "stem_type": stem_type,
            "stem_w": stem_w,
            "block_type": block_type,
            "depths": ds,
            "widths": ws,
            "strides": ss,
            "bot_muls": bs,
            "group_ws": gs,
            "head_w": head_w,
            "se_r": se_r,
            "num_classes": num_classes,
        }

    def __init__(self, w_a, w_0, w_m, d, group_w, stride=2, bot_mul=1.0, stem_type="simple_stem_in", stem_w=32,
                 block_type="res_bottleneck_block", head_w=0, num_classes=1000, se_r=0.0, in_channels=3):
        params = RegNet.regnet_get_params(w_a, w_0, w_m, d, stride, bot_mul, group_w, stem_type, stem_w, block_type,
                                          head_w, num_classes, se_r)
        print(params)
        super(RegNet, self).__init__(params["depths"], params["stem_type"], params["stem_w"], params["block_type"],
                                     params["widths"], params["strides"], params["bot_muls"], params["group_ws"],
                                     params["head_w"], params["num_classes"], params["se_r"], in_channels)


@register_model
def regnet_x_200mf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_x_200mf"]
    model = RegNet(36.44, 24, 2.49, 13, 8, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_x_400mf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_x_400mf"]
    model = RegNet(24.48, 24, 2.54, 22, 16, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_x_600mf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_x_600mf"]
    model = RegNet(36.97, 48, 2.24, 16, 24, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_x_800mf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_x_800mf"]
    model = RegNet(35.73, 56, 2.28, 16, 16, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_x_1_6gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_x_1_6gf"]
    model = RegNet(34.01, 80, 2.25, 18, 24, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_x_3_2gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_x_3_2gf"]
    model = RegNet(26.31, 88, 2.25, 25, 48, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_x_4_0gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_x_4_0gf"]
    model = RegNet(38.65, 96, 2.43, 23, 40, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_x_6_4gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_x_6_4gf"]
    model = RegNet(60.83, 184, 2.07, 17, 56, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_x_8_0gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_x_8_0gf"]
    model = RegNet(49.56, 80, 2.88, 23, 120, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_x_12gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_x_12gf"]
    model = RegNet(73.36, 168, 2.37, 19, 112, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_x_16gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_x_16gf"]
    model = RegNet(55.59, 216, 2.1, 22, 128, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_x_32gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_x_32gf"]
    model = RegNet(69.86, 320, 2.0, 23, 168, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_y_200mf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_y_200mf"]
    model = RegNet(36.44, 24, 2.49, 13, 8, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_y_400mf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_y_400mf"]
    model = RegNet(27.89, 48, 2.09, 16, 8, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_y_600mf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_y_600mf"]
    model = RegNet(32.54, 48, 2.32, 15, 16, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_y_800mf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_y_800mf"]
    model = RegNet(38.84, 56, 2.4, 14, 16, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_y_1_6gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_y_1_6gf"]
    model = RegNet(20.71, 48, 2.65, 27, 24, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_y_3_2gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_y_3_2gf"]
    model = RegNet(42.63, 80, 2.66, 21, 24, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_y_4_0gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_y_4_0gf"]
    model = RegNet(31.41, 96, 2.24, 22, 64, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_y_6_4gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_y_6_4gf"]
    model = RegNet(33.22, 112, 2.27, 25, 72, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_y_8_0gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_y_8_0gf"]
    model = RegNet(76.82, 192, 2.19, 17, 56, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_y_12gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_y_12gf"]
    model = RegNet(73.36, 168, 2.37, 19, 112, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_y_16gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_y_16gf"]
    model = RegNet(106.23, 200, 2.48, 18, 112, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnet_y_32gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["regnet_y_32gf"]
    model = RegNet(115.89, 232, 2.53, 20, 232, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model
