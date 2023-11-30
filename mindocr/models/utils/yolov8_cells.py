import math
from copy import deepcopy

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.common import initializer as init


class YOLOv8BaseConfig:
    def __init__(self,
                 nc=5,
                 reg_max=16,
                 stride=None,
                 depth_multiple=1.0,
                 width_multiple=1.0,
                 max_channels=1024,
                 sync_bn=False,
                 ):
        if stride is None:
            stride = [8, 16, 32, 64]
        self.model_name = 'yolov8'
        self.nc = nc
        self.reg_max = reg_max
        self.stride = stride
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        self.max_channels = max_channels
        self.sync_bn = sync_bn


class Upsample(nn.Cell):
    """
    Using the interpolate method specified by `mode` resize the input tensor.

    Args:
        scales (tuple[float], optional): a tuple of float. Describe the scale along each dimension.
            Its length is the same as that of shape of `x`. The numbers in `scales` must all be positive. Only one of
            `scales` and `sizes` can be specified.
        sizes (tuple[int], optional): a tuple of int, describes the shape of the output tensor. The numbers in `sizes`
            must all be positive. Only one of `scales` and `sizes` can be specified.  If `sizes` is specified, then set
            `scales` to 'None' in this operator's input list. It is 1 int elements :math:`(new_width,)` when `mode`
            is "linear". It is 2 int elements :math:`(new_height, new_width)` when `mode` is "bilinear".
        mode (string): The method used to interpolate: 'linear', 'bilinear' and 'nearest'. Default is 'nearest'.
    """

    def __init__(self, sizes=None, scales=None, mode="nearest"):
        super(Upsample, self).__init__()
        self.sizes = sizes
        self.scales = scales
        self.mode = mode

    def construct(self, x):
        if self.mode == "nearest" and self.scales:
            size = (int(x.shape[-2] * self.scales + 0.5), int(x.shape[-1] * self.scales + 0.5))
            return ops.ResizeNearestNeighbor(size)(x)
        else:
            return ops.interpolate(x, size=self.sizes, scale_factor=self.scales, mode=self.mode)


def meshgrid(inputs, indexing="xy"):
    # An alternative implementation of ops.meshgrid, Only supports inputs with a length of 2.
    # Meshgrid op is not supported on a specific model of machine an alternative
    # solution is adopted, which will be updated later.
    x, y = inputs
    nx, ny = x.shape[0], y.shape[0]
    xv, yv = None, None
    if indexing == "xy":
        xv = ops.tile(x.view(1, -1), (ny, 1))
        yv = ops.tile(y.view(-1, 1), (1, nx))
    elif indexing == "ij":
        xv = ops.tile(x.view(-1, 1), (1, ny))
        yv = ops.tile(y.view(1, -1), (nx, 1))

    return xv, yv


class DFL(nn.Cell):
    # Integral module of Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
    # https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, has_bias=False)
        self.conv.weight.requires_grad = False
        self.c1 = c1
        self.softmax = ops.Softmax(axis=1)

    def construct(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        x = self.softmax(x.view(b, 4, self.c1, a).swapaxes(2, 1))
        x = self.conv(x)
        x = x.view(b, 4, a)
        return x

    def initialize_conv_weight(self):
        import numpy as np
        self.conv.weight = ops.assign(
            self.conv.weight, Tensor(np.arange(self.c1).reshape((1, self.c1, 1, 1)), dtype=ms.float32)
        )


class Concat(nn.Cell):
    """
    Connect tensor in the specified axis.
    """

    def __init__(self, axis=1):
        super(Concat, self).__init__()
        self.axis = axis

    def construct(self, x):
        return ops.concat(x, self.axis)


class Bottleneck(nn.Cell):
    # Standard bottleneck
    def __init__(
            self, c1, c2, shortcut=True, k=(1, 3), g=(1, 1), e=0.5, act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, k[0], 1, g=g[0], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c_, c2, k[1], 1, g=g[1], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.add = shortcut and c1 == c2

    def construct(self, x):
        if self.add:
            out = x + self.conv2(self.conv1(x))
        else:
            out = self.conv2(self.conv1(x))
        return out


class C2f(nn.Cell):
    # CSP Bottleneck with 2 convolutions
    def __init__(
            self, c1, c2, n=1, shortcut=False, g=1, e=0.5, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        _c = int(c2 * e)  # hidden channels
        self.cv1 = ConvNormAct(c1, 2 * _c, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.cv2 = ConvNormAct(
            (2 + n) * _c, c2, 1, momentum=momentum, eps=eps, sync_bn=sync_bn
        )  # optional act=FReLU(c2)
        self.m = nn.CellList(
            [
                Bottleneck(_c, _c, shortcut, k=(3, 3), g=(1, g), e=1.0, momentum=momentum, eps=eps, sync_bn=sync_bn)
                for _ in range(n)
            ]
        )

    def construct(self, x):
        y = ()
        x = self.cv1(x)
        _c = x.shape[1] // 2
        x_tuple = ops.split(x, axis=1, split_size_or_sections=_c)
        y += x_tuple
        for i in range(len(self.m)):
            m = self.m[i]
            out = m(y[-1])
            y += (out,)

        return self.cv2(ops.concat(y, axis=1))


class SPPF(nn.Cell):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(
            self, c1, c2, k=5, act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # equivalent to SPP(k=(5, 9, 13))
        super(SPPF, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, 1, 1, act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c_ * 4, c2, 1, 1, act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.concat = ops.Concat(axis=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, pad_mode="same")

    def construct(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        y = self.conv2(self.concat((x, y1, y2, y3)))
        return y


class Identity(nn.Cell):
    def construct(self, x):
        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    if isinstance(p, list):
        assert len(p) == 2
        p = (p[0], p[0], p[1], p[1])
    return p


class ConvNormAct(nn.Cell):
    """Conv2d + BN + Act

    Args:
        c1 (int): In channels, the channel number of the input tensor of the Conv2d layer.
        c2 (int): Out channels, the channel number of the output tensor of the Conv2d layer.
        k (Union[int, tuple[int]]): Kernel size, Specifies the height and width of the 2D convolution kernel.
            The data type is an integer or a tuple of two integers. An integer represents the height
            and width of the convolution kernel. A tuple of two integers represents the height
            and width of the convolution kernel respectively. Default: 1.
        s (Union[int, tuple[int]]): Stride, the movement stride of the 2D convolution kernel.
            The data type is an integer or a tuple of two integers. An integer represents the movement step size
            in both height and width directions. A tuple of two integers represents the movement step size in the height
            and width directions respectively. Default: 1.
        p (Union[None, int, tuple[int]]): Padding, the number of padding on the height and width directions of the
            input.
            The data type is None or an integer or a tuple of four integers. If `padding` is an None, then padding
            with autopad.
            If `padding` is an integer, then the top, bottom, left, and right padding are all equal to `padding`.
            If `padding` is a tuple of 4 integers, then the top, bottom, left, and right padding
            is equal to `padding[0]`, `padding[1]`, `padding[2]`, and `padding[3]` respectively.
            The value should be greater than or equal to 0. Default: None.
        g (int): Group, Splits filter into groups, `c1` and `c2` must be
            divisible by `group`. If the group is equal to `c1` and `c2`,
            this 2D convolution layer also can be called 2D depthwise convolution layer. Default: 1.
        d (Union[int, tuple[int]]): Dilation, Dilation size of 2D convolution kernel.
            The data type is an integer or a tuple of two integers. If :math:`k > 1`, the kernel is sampled
            every `k` elements. The value of `k` on the height and width directions is in range of [1, H]
            and [1, W] respectively. Default: 1.
        act (Union[bool, nn.Cell]): Activation. The data type is bool or nn.Cell. If `act` is True,
            then the activation function uses nn.SiLU. If `act` is False, do not use activation function.
            If 'act' is nn.Cell, use the object of this cell as the activation function. Default: True.
        sync_bn (bool): Whether the BN layer use nn.SyncBatchNorm. Default: False.
    """

    def __init__(
            self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvNormAct, self).__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, pad_mode="pad", padding=autopad(k, p, d), group=g, dilation=d, has_bias=False
        )

        if sync_bn:
            self.bn = nn.SyncBatchNorm(c2, momentum=momentum, eps=eps)
        else:
            self.bn = nn.BatchNorm2d(c2, momentum=momentum, eps=eps)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Cell) else Identity)

    def construct(self, x):
        return self.act(self.bn(self.conv(x)))


def initialize_default(model):
    for _, cell in model.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(
                init.initializer(init.HeUniform(negative_slope=math.sqrt(5)), cell.weight.shape, cell.weight.dtype)
            )
            if cell.bias is not None:
                fan_in, _ = _calculate_fan_in_and_fan_out(cell.weight.shape)
                bound = 1 / math.sqrt(fan_in)
                cell.bias.set_data(init.initializer(init.Uniform(bound), cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Dense):
            cell.weight.set_data(
                init.initializer(init.HeUniform(negative_slope=math.sqrt(5)), cell.weight.shape, cell.weight.dtype)
            )
            if cell.bias is not None:
                fan_in, _ = _calculate_fan_in_and_fan_out(cell.weight.shape)
                bound = 1 / math.sqrt(fan_in)
                cell.bias.set_data(init.initializer(init.Uniform(bound), cell.bias.shape, cell.bias.dtype))


def _calculate_fan_in_and_fan_out(shape):
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = shape[1]
    num_output_fmaps = shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


class Model(nn.Cell):
    def __init__(self, model_cfg, in_channels=3):
        super(Model, self).__init__()
        self.model, self.save, self.layers_param = parse_model(
            deepcopy(model_cfg), ch=[in_channels]
        )
        # Recompute
        if hasattr(model_cfg, "recompute") and model_cfg.recompute and model_cfg.recompute_layers > 0:
            for i in range(model_cfg.recompute_layers):
                self.model[i].recompute()

    def construct(self, x):
        y = ()  # outputs
        for i in range(len(self.model)):
            m = self.model[i]
            iol, f, _, _ = self.layers_param[i]  # iol: index of layers

            if not (isinstance(f, int) and f == -1):  # if not from previous layer
                if isinstance(f, int):
                    x = y[f]
                else:
                    _x = ()
                    for j in f:
                        if j == -1:
                            _x += (x,)
                        else:
                            _x += (y[j],)
                    x = _x

            x = m(x)  # run
            y += (x,)  # save output
        return y


def parse_model(d, ch):  # model_dict, input_channels(3)
    _SYNC_BN = d.sync_bn
    max_channels = d.max_channels
    gd, gw = d.depth_multiple, d.width_multiple

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    layers_param = []
    num_total_param, num_train_param = 0, 0
    for i, (f, n, m, args) in enumerate(d.backbone):  # from, number, module, args
        kwargs = {}
        if isinstance(m, str):
            if m == "ConvNormAct":
                m = ConvNormAct
            elif m == "C2f":
                m = C2f
            elif m == "SPPF":
                m = SPPF
            elif m == "Upsample":
                m = Upsample
            elif m == "Concat":
                m = Concat
            else:
                raise ValueError("{} is not supported.".format(m))

        _args = []
        for a in args:
            if a == "None":
                a = None
            _args.append(a)
        args = _args

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in (
                nn.Conv2d,
                ConvNormAct,
                SPPF,
                C2f,
        ):
            c1, c2 = ch[f], args[0]
            if max_channels:
                c2 = min(c2, max_channels)
            c2 = math.ceil(c2 * gw / 8) * 8

            args = [c1, c2, *args[1:]]
            if m in (
                    ConvNormAct,
                    SPPF,
                    C2f,
            ):
                kwargs["sync_bn"] = _SYNC_BN
            if m in (C2f,):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m in (nn.BatchNorm2d, nn.SyncBatchNorm):
            args = [ch[f]]
        elif m in (Concat,):
            c2 = sum([ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.SequentialCell([m(*args, **kwargs) for _ in range(n)]) if n > 1 else m(*args, **kwargs)

        t = str(m)  # module type
        np = sum([x.size for x in m_.get_parameters()])  # number params
        np_trainable = sum([x.size for x in m_.trainable_params()])  # number trainable params
        num_total_param += np
        num_train_param += np_trainable
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        layers_param.append((i, f, t, np))
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.CellList(layers), sorted(save), layers_param
