import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

__all__ = ["MultiAspectGCAttention"]


class _LayerNorm(nn.Cell):
    """A temp replacement of nn.LayerNorm([normalized_shape, 1, 1], 1, 1)"""
    def __init__(self, normalized_shape):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def construct(self, x: Tensor):
        x = x.reshape((x.shape[0], -1))
        x = self.layer_norm(x)
        x = x[..., None, None]
        return x


class MultiAspectGCAttention(nn.Cell):
    def __init__(
        self,
        inplanes: int,
        ratio: float,
        headers: int,
        pooling_type: str = "att",
        att_scale: bool = False,
        fusion_type: str = "channel_add",
    ) -> None:
        super(MultiAspectGCAttention, self).__init__()
        assert pooling_type in ["avg", "att"]

        assert fusion_type in ["channel_add", "channel_mul", "channel_concat"]
        assert (
            inplanes % headers == 0 and inplanes >= 8
        )  # inplanes must be divided by headers evenly

        self.headers = headers
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_type = fusion_type
        self.att_scale = att_scale

        self.single_header_inplanes = int(inplanes / headers)

        if pooling_type == "att":
            self.conv_mask = nn.Conv2d(
                self.single_header_inplanes, 1, kernel_size=1, has_bias=True
            )
            self.softmax = ops.Softmax(axis=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if fusion_type == "channel_add":
            self.channel_add_conv = nn.SequentialCell(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1, has_bias=True),
                _LayerNorm([self.planes]),
                nn.ReLU(),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1, has_bias=True),
            )
        elif fusion_type == "channel_concat":
            self.channel_concat_conv = nn.SequentialCell(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1, has_bias=True),
                _LayerNorm([self.planes]),
                nn.ReLU(),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1, has_bias=True),
            )
            # for concat
            self.cat_conv = nn.Conv2d(
                2 * self.inplanes, self.inplanes, kernel_size=1, has_bias=True
            )
            self.layer_norm = ops.LayerNorm(begin_norm_axis=1, begin_params_axis=1)
        elif fusion_type == "channel_mul":
            self.channel_mul_conv = nn.SequentialCell(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1, has_bias=True),
                _LayerNorm([self.planes]),
                nn.ReLU(),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1, has_bias=True),
            )

    def spatial_pool(self, x: Tensor) -> Tensor:
        N, _, H, W = x.shape
        if self.pooling_type == "att":
            # [N*headers, C', H , W] C = headers * C'
            x = x.reshape(N * self.headers, self.single_header_inplanes, H, W)
            input_x = x

            # [N*headers, C', H * W] C = headers * C'
            input_x = input_x.reshape(
                N * self.headers, self.single_header_inplanes, H * W
            )

            # [N*headers, 1, C', H * W]
            input_x = ops.expand_dims(input_x, 1)
            # [N*headers, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N*headers, 1, H * W]
            context_mask = context_mask.reshape(N * self.headers, 1, H * W)

            # scale variance
            if self.att_scale and self.headers > 1:
                context_mask = context_mask / ops.sqrt(self.single_header_inplanes)

            # [N*headers, 1, H * W]
            context_mask = self.softmax(context_mask)

            # [N*headers, 1, H * W, 1]
            context_mask = ops.expand_dims(context_mask, -1)
            # [N*headers, 1, C', 1] = [N*headers, 1, C', H * W] * [N*headers, 1, H * W, 1]
            context = ops.matmul(input_x, context_mask)

            # [N, headers * C', 1, 1]
            context = context.reshape(
                N, self.headers * self.single_header_inplanes, 1, 1
            )
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def construct(self, x: Tensor) -> Tensor:
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x

        if self.fusion_type == "channel_mul":
            # [N, C, 1, 1]
            channel_mul_term = ops.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        elif self.fusion_type == "channel_add":
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        else:
            # [N, C, 1, 1]
            channel_concat_term = self.channel_concat_conv(context)

            # use concat
            _, _, H, W = out.shape

            channel_concat_term = ops.tile(channel_concat_term, (1, 1, H, W))

            out = ops.concat([out, channel_concat_term], axis=1)
            out = self.cat_conv(out)
            gamma = ops.ones(out.shape[1:], out.dtype)
            beta = ops.zeros(out.shape[1:], out.dtype)
            out, _, _ = self.layer_norm(out, gamma, beta)
            out = ops.relu(out)

        return out
