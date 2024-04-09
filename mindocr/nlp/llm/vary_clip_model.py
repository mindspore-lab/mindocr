from collections import OrderedDict

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops

from mindocr.nlp.utils.layers import LayerNorm, Linear


class QuickGELU(nn.Cell):
    def construct(self, x: Tensor):
        return x * ops.sigmoid(1.702 * x)


class CLIPAttention(nn.Cell):
    """Multi-head attention module for CLIP"""

    def __init__(self, embed_dim, num_heads, param_init_type=ms.float32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.k_proj = Linear(self.embed_dim, self.embed_dim, param_init_type=param_init_type)
        self.v_proj = Linear(self.embed_dim, self.embed_dim, param_init_type=param_init_type)
        self.q_proj = Linear(self.embed_dim, self.embed_dim, param_init_type=param_init_type)
        self.out_proj = Linear(self.embed_dim, self.embed_dim, param_init_type=param_init_type)
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.batch_matmul = ops.BatchMatMul()
        self.batch_matmul_q_k = ops.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax()

    def construct(self, x):
        bsz, tgt_len, embed_dim = x.shape
        query_states = self.transpose(
            self.reshape(self.q_proj(x) * self.scale, (bsz, tgt_len, self.num_heads, self.head_dim)), (0, 2, 1, 3)
        )
        key_states = self.transpose(
            self.reshape(self.k_proj(x), (bsz, tgt_len, self.num_heads, self.head_dim)), (0, 2, 1, 3)
        )
        value_states = self.transpose(
            self.reshape(self.v_proj(x), (bsz, tgt_len, self.num_heads, self.head_dim)), (0, 2, 1, 3)
        )

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self.reshape(query_states, proj_shape)
        key_states = self.reshape(key_states, proj_shape)
        value_states = self.reshape(value_states, proj_shape)

        src_len = tgt_len
        attn_weights = self.batch_matmul_q_k(query_states, key_states)
        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )
        attn_weights = self.softmax(attn_weights)
        attn_output = self.batch_matmul(attn_weights, value_states)
        if attn_output.shape != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = self.reshape(attn_output, (bsz, self.num_heads, tgt_len, self.head_dim))
        attn_output = self.transpose(attn_output, (0, 2, 1, 3))
        attn_output = self.reshape(attn_output, (bsz, tgt_len, embed_dim))

        attn_output = self.out_proj(attn_output)

        return attn_output


class ResidualAttentionBlock(nn.Cell):
    """ResidualAttention module for CLIP"""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: Tensor = None,
        param_init_type=ms.float32,
        ln_param_init_type=ms.float32,
    ):
        super().__init__()

        self.attn = CLIPAttention(d_model, n_head, param_init_type=param_init_type)
        self.ln_1 = LayerNorm((d_model,), eps=1e-5, param_init_type=ln_param_init_type)
        self.mlp = nn.SequentialCell(
            OrderedDict(
                [
                    ("c_fc", Linear(d_model, d_model * 4, param_init_type=param_init_type)),
                    ("gelu", QuickGELU()),
                    ("c_proj", Linear(d_model * 4, d_model, param_init_type=param_init_type)),
                ]
            )
        )
        self.ln_2 = LayerNorm((d_model,), eps=1e-5, param_init_type=ln_param_init_type)
        self.attn_mask = Parameter(attn_mask) if attn_mask is not None else None

    def construct(self, x: Tensor):
        residual0 = x
        x_type = x.dtype
        x = self.ln_1(x.to(ms.float32)).to(x_type)
        x = residual0 + self.attn(x)
        residual1 = x
        x = self.ln_2(x.to(ms.float32)).to(x_type)
        x = residual1 + self.mlp(x)
        return x


class Transformer(nn.Cell):
    """Vision Transformer module for CLIP"""

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: Tensor = None,
        param_init_type=ms.float32,
        ln_param_init_type=ms.float32,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.CellList(
            [
                ResidualAttentionBlock(
                    width, heads, attn_mask, param_init_type=param_init_type, ln_param_init_type=ln_param_init_type
                )
                for _ in range(layers)
            ]
        )

    def construct(self, x: Tensor):
        encoder_states = ()
        hidden_state = x
        for block in self.resblocks:
            encoder_states += (hidden_state,)
            hidden_state = block(hidden_state)
        encoder_states += (hidden_state,)
        return encoder_states


class VisionTransformer(nn.Cell):
    """CLIP module for Vary system"""

    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        vision_select_layer: int,
        param_init_type=ms.float32,
        ln_param_init_type=ms.float32,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.vision_select_layer = vision_select_layer
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            has_bias=False,
            pad_mode="pad",
            weight_init="uniform",
            bias_init="uniform",
            dtype=param_init_type,
        )

        scale = width**-0.5
        self.class_embedding = Parameter((scale * ops.randn(width)).astype(param_init_type))
        self.positional_embedding = Parameter(
            (scale * ops.randn(((input_resolution // patch_size) ** 2 + 1, width))).astype(param_init_type)
        )
        self.ln_pre = LayerNorm((width,), eps=1e-5, param_init_type=ln_param_init_type)
        self.transformer = Transformer(
            width, layers, heads, param_init_type=param_init_type, ln_param_init_type=ln_param_init_type
        )

    def construct(self, x: Tensor):
        x_type = x.dtype
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape((x.shape[0], x.shape[1], -1))  # shape = [*, width, grid**2]
        x = x.permute((0, 2, 1))  # shape = [*, grid**2, width]
        x = ops.cat(
            [self.class_embedding.to(x_type) + ops.zeros((x.shape[0], 1, x.shape[-1]), dtype=x_type), x], axis=1
        )  # shape = [*, grid**2 + 1, width]
        x = x + self.positional_embedding.to(x_type)  # torch version: CLIPVisionEmbeddings
        x = self.ln_pre(x)  # modeling_clip.py L842
        x = self.transformer(x)  # modeling_clip.py L844, error 1e-3
        x = x[self.vision_select_layer][:, 1:]
        return x


def build_model(param_init_type=ms.float32, ln_param_init_type=ms.float32):
    """construct the CLIP module and load ckpt"""
    vision_width = 1024
    vision_layers = 24
    vision_patch_size = 14
    grid_size = round(256**0.5)
    image_resolution = vision_patch_size * grid_size
    out_width = 1024
    model = VisionTransformer(
        input_resolution=image_resolution,  # image_size in transformers
        patch_size=vision_patch_size,  # patch_size in transformers
        width=vision_width,  # hidden_size
        layers=vision_layers,  # num_hidden_layers
        heads=grid_size,  # num_attention_heads
        output_dim=out_width,  # projection_dim in transformers, default: 1024
        vision_select_layer=-2,
        param_init_type=param_init_type,
        ln_param_init_type=ln_param_init_type,
    )

    return model
