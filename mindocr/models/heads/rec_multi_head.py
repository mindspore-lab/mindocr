from typing import Type, Union

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import Constant, TruncatedNormal, initializer

from .rec_ctc_head import CTCHead
from .rec_sar_head import SARHead

__all__ = ["MultiHead"]


class Swish(nn.Cell):
    def __init__(self):
        super().__init__()
        self.result = None
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        result = x * self.sigmoid(x)
        return result


class ConvBNLayer(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple] = 3,
        stride: int = 1,
        padding: Union[int, tuple] = 0,
        bias_attr: bool = False,
        groups: int = 1,
        act: Union[Type[nn.Cell], str] = nn.GELU,
    ) -> None:
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode="pad",
            padding=padding,
            group=groups,
            has_bias=bias_attr,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        if act == "Swish":
            self.act = Swish()
        else:
            self.act = act()

    def construct(self, inputs: Tensor) -> Tensor:
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class Im2Seq(nn.Cell):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def construct(self, x):
        B, C, H, W = x.shape
        assert H == 1
        x = x.squeeze(axis=2)
        x = x.transpose([0, 2, 1])  # (NTC)(batch, width, channels)
        return x


class Attention(nn.Cell):
    def __init__(self,
                 dim,
                 num_heads=8,
                 mixer='Global',
                 HW=None,
                 local_k=[7, 11],
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)
        self.softmax = nn.Softmax(axis=-1)
        self.HW = HW
        if HW is not None:
            H = HW[0]
            W = HW[1]
            self.N = H * W
            self.C = dim
        if mixer == 'Local' and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = ops.ones([H * W, H + hk - 1, W + wk - 1], dtype=ms.float32)
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h:h + hk, w:w + wk] = 0.
            mask_paddle = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk //
                               2].flatten(1)
            mask_inf = ops.full([H * W, H * W], '-inf', dtype=ms.float32)
            mask = ops.where(mask_paddle < 1, mask_paddle, mask_inf)
            self.mask = mask.unsqueeze([0, 1])
        self.mixer = mixer

    def construct(self, x):
        bs = x.shape[0]
        qkv = self.qkv(x).reshape(
            (bs, -1, 3, self.num_heads, self.head_dim)).transpose(
                (2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q.matmul(k.transpose((0, 1, 3, 2))))
        if self.mixer == 'Local':
            attn += self.mask
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((bs, -1, self.dim))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ConvMixer(nn.Cell):
    def __init__(
            self,
            dim,
            num_heads=8,
            HW=[8, 25],
            local_k=[3, 3], ):
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2d(
            dim,
            dim,
            local_k,
            1, [local_k[0] // 2, local_k[1] // 2],
            groups=num_heads,
            weight_init='HeNormal')

    def construct(self, x):
        h = self.HW[0]
        w = self.HW[1]
        x = x.transpose([0, 2, 1]).reshape([0, self.dim, h, w])
        x = self.local_mixer(x)
        x = x.flatten(2).transpose([0, 2, 1])
        return x


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = Tensor(1 - drop_prob, dtype=x.dtype)
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + ops.rand(shape, dtype=x.dtype)
    random_tensor = ops.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def construct(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Cell):
    def __init__(self):
        super(Identity, self).__init__()

    def construct(self, input):
        return input


class Mlp(nn.Cell):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='Swish',
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = Swish()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(p=drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Cell):
    def __init__(self,
                 dim,
                 num_heads,
                 mixer='Global',
                 local_mixer=[7, 11],
                 HW=None,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer='Swish',
                 # norm_layer='nn.LayerNorm',
                 epsilon=1e-6,
                 prenorm=True):
        super().__init__()
        self.norm1 = nn.LayerNorm((dim,))
        if mixer == 'Global' or mixer == 'Local':
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop)
        elif mixer == 'Conv':
            self.mixer = ConvMixer(
                dim, num_heads=num_heads, HW=HW, local_k=local_mixer)
        else:
            raise TypeError("The mixer must be one of [Global, Local, Conv]")

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = nn.LayerNorm((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.prenorm = prenorm

    def construct(self, x):
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class EncoderWithSVTR(nn.Cell):
    def __init__(
            self,
            in_channels,
            dims=64,  # XS
            depth=2,
            hidden_dims=120,
            use_guide=False,
            num_heads=8,
            qkv_bias=True,
            mlp_ratio=2.0,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path=0.,
            kernel_size=(3, 3),
            qk_scale=None):
        super(EncoderWithSVTR, self).__init__()
        self.depth = depth
        self.use_guide = use_guide
        self.conv1 = ConvBNLayer(
            in_channels,
            in_channels // 8,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2),
            act='Swish')
        self.conv2 = ConvBNLayer(
            in_channels // 8, hidden_dims, kernel_size=1, act='Swish')

        self.svtr_block = nn.CellList([
            Block(
                dim=hidden_dims,
                num_heads=num_heads,
                mixer='Global',
                HW=None,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer='Swish',
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                # norm_layer='nn.LayerNorm',
                epsilon=1e-05,
                prenorm=False) for i in range(depth)
        ])
        self.norm = nn.LayerNorm((hidden_dims,), epsilon=1e-6)
        self.conv3 = ConvBNLayer(
            hidden_dims, in_channels, kernel_size=1, act='Swish')
        # last conv-nxn, the input is concat of input tensor and conv3 output tensor
        self.conv4 = ConvBNLayer(
            2 * in_channels,
            in_channels // 8,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2),
            act='Swish')

        self.conv1x1 = ConvBNLayer(
            in_channels // 8, dims, kernel_size=1, act='Swish')
        self.out_channels = dims
        self.apply(self._init_weights)

    def _init_weights(self, m):
        trunc_normal_ = TruncatedNormal(sigma=0.02)
        zeros_ = Constant(value=0.0)
        ones_ = Constant(value=1.0)
        if isinstance(m, nn.Dense):
            # trunc_normal_(m.weight)
            m.weight = Parameter(initializer(trunc_normal_, m.weight.shape, ms.float32))
            if isinstance(m, nn.Dense) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.beta)
            ones_(m.gamma)

    def construct(self, x):
        # for use guide
        if self.use_guide:
            z = x
            z = ops.stop_gradient(z)
        else:
            z = x
        # for short cut
        h = z
        # reduce dim
        z = self.conv1(z)
        z = self.conv2(z)
        # SVTR global block
        B, C, H, W = z.shape
        z = z.flatten(order='C', start_dim=2).transpose([0, 2, 1])
        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z)
        # last stage
        z = z.reshape([-1, H, W, C]).transpose([0, 3, 1, 2])
        z = self.conv3(z)
        z = ops.concat((h, z), axis=1)
        z = self.conv1x1(self.conv4(z))
        return z


class SequenceEncoder(nn.Cell):
    def __init__(self, in_channels, encoder_type, hidden_size=48, **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        self.encoder_type = encoder_type
        if encoder_type == 'reshape':
            self.only_reshape = True
        else:
            support_encoder_dict = {
                'reshape': Im2Seq,
                'svtr': EncoderWithSVTR,
            }
            assert encoder_type in support_encoder_dict, '{} must in {}'.format(
                encoder_type, support_encoder_dict.keys())
            if encoder_type == "svtr":
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, **kwargs)
            else:
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, hidden_size)
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def construct(self, x):
        if self.encoder_type != 'svtr':
            x = self.encoder_reshape(x)
            if not self.only_reshape:
                x = self.encoder(x)
            return x
        else:
            x = self.encoder(x)
            x = self.encoder_reshape(x)
            return x


class MultiHead(nn.Cell):
    def __init__(self, in_channels, out_channels_list, **kwargs):
        super().__init__()
        self.head_list = kwargs.pop('head_list')
        out_channels_dict = {}
        for out_channels in out_channels_list:
            out_channels_dict.update(out_channels)

        self.gtc_head = 'sar'
        assert len(self.head_list) >= 2
        for idx, head_name in enumerate(self.head_list):
            name = list(head_name)[0]
            if name == 'SARHead':
                # sar head
                sar_args = self.head_list[idx][name]
                self.sar_head = SARHead(in_channels=in_channels,
                                        out_channels=out_channels_dict['SARLabelDecode'], **sar_args)
            elif name == 'CTCHead':
                # ctc neck
                self.encoder_reshape = Im2Seq(in_channels)
                neck_args = self.head_list[idx][name]['Neck']
                encoder_type = neck_args.pop('name')
                self.ctc_encoder = SequenceEncoder(in_channels=in_channels,
                                                   encoder_type=encoder_type, **neck_args)
                # ctc head
                head_args = self.head_list[idx][name]['Head']
                self.ctc_head = CTCHead(in_channels=self.ctc_encoder.out_channels,
                                        out_channels=out_channels_dict['CTCLabelDecode'], **head_args)
            else:
                raise NotImplementedError(
                    '{} is not supported in MultiHead yet'.format(name))

    def construct(self, x, targets=None):
        ctc_encoder = self.ctc_encoder(x)
        ctc_out = self.ctc_head(ctc_encoder)

        # eval mode
        if not self.training:
            ctc_out = ctc_out.transpose((1, 0, 2))
            return ctc_out

        sar_out = self.sar_head(x, targets[1:])
        head_out = (ctc_out, sar_out,)
        return head_out
