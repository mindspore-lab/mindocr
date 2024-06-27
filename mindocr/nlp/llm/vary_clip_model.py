from collections import OrderedDict

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import initializer

from mindocr.nlp.utils.layers import LayerNorm, Linear


class LoraAdapter(nn.Cell):
    def __init__(
        self,
        d_model,
        out_feat,
        r=16,
        param_init_type=ms.float32,
        compute_dtype=ms.float32,
    ):
        super().__init__()
        self.d_model = d_model
        self.out_feat = out_feat

        self.lora_a = Linear(
            self.d_model,
            r,
            has_bias=False,
            param_init_type=param_init_type,
            compute_dtype=compute_dtype,
        )
        self.lora_b = Linear(
            r,
            self.out_feat,
            has_bias=False,
            param_init_type=param_init_type,
            compute_dtype=compute_dtype,
        )

    def construct(self, x):
        down = self.lora_a(x)
        up = self.lora_b(down)
        output = up
        return output


class QuickGELU(nn.Cell):
    def construct(self, x: Tensor):
        return x * ops.sigmoid(1.702 * x)


class VisualAttention(nn.Cell):
    """self-attention layer class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self, embed_dim, num_heads, lora_repeat_num=4, param_init_type=ms.float32, compute_dtype=ms.float32
    ):
        super(VisualAttention, self).__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = embed_dim // num_heads
        self.num_attention_heads_per_partition = num_heads
        self.hidden_size_per_partition = embed_dim

        # Strided linear layer.
        self.in_proj = Linear(embed_dim, 3 * embed_dim, param_init_type=param_init_type, compute_dtype=compute_dtype)
        self.in_proj_lora = []
        for _ in range(lora_repeat_num):
            self.in_proj_lora.append(LoraAdapter(
                d_model=embed_dim,
                out_feat=3 * embed_dim,
                param_init_type=param_init_type,
                compute_dtype=compute_dtype,
            ))
        self.in_proj_lora = nn.CellList(self.in_proj_lora)

        self.out_proj = Linear(embed_dim, embed_dim, param_init_type=param_init_type, compute_dtype=compute_dtype)
        self.out_proj_lora = []
        for _ in range(lora_repeat_num):
            self.out_proj_lora.append(LoraAdapter(
                d_model=embed_dim,
                out_feat=embed_dim,
                param_init_type=param_init_type,
                compute_dtype=compute_dtype,
            ))
        self.out_proj_lora = nn.CellList(self.out_proj_lora)
        self.norm_factor = self.hidden_size_per_attention_head**0.5

    def construct(self, query, idx=None):
        # query/key/value: [sq, b, h]
        sq, b, _ = query.shape

        sk = sq
        mixed_x_layer = self.in_proj(query)
        if idx is not None:
            lora_res = self.in_proj_lora[idx](query)
            mixed_x_layer += lora_res

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.shape[:-1] + \
            (self.num_attention_heads_per_partition,
             3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        query_layer, key_layer, value_layer = mixed_x_layer.split(self.hidden_size_per_attention_head, axis=3)

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            sq, b * self.num_attention_heads_per_partition, self.hidden_size_per_attention_head
        ).transpose(1, 0, 2)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(
            sk, b * self.num_attention_heads_per_partition, self.hidden_size_per_attention_head
        ).transpose(1, 0, 2)

        q_scaled = query_layer / self.norm_factor
        attention_probs = ops.BatchMatMul(transpose_b=True)(q_scaled, key_layer)
        attention_probs = ops.softmax(attention_probs, axis=-1)

        value_layer = value_layer.view(
            sk, b * self.num_attention_heads_per_partition, self.hidden_size_per_attention_head
        ).transpose(1, 0, 2)

        # matmul: [b * np, sq, hn]
        context_layer = ops.bmm(attention_probs, value_layer)

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(
            b, self.num_attention_heads_per_partition, sq, self.hidden_size_per_attention_head
        )

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3)

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.shape[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.out_proj(context_layer)
        if idx is not None:
            lora_res = self.out_proj_lora[idx](context_layer)
            output += lora_res

        return output


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
        compute_dtype=ms.float32,
        layernorm_compute_type=ms.float32,
        mlp_ratio=4.0,
        lora_repeat_num=4,
        model_type="clip",
    ):
        super().__init__()

        self.use_clip = model_type == "clip"
        if self.use_clip:
            self.attn = CLIPAttention(d_model, n_head, param_init_type=param_init_type)
        else:
            self.attn = VisualAttention(d_model, n_head, param_init_type=param_init_type, compute_dtype=compute_dtype)
        self.ln_1 = LayerNorm((d_model,), eps=1e-6, param_init_type=ln_param_init_type)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.SequentialCell(
            OrderedDict(
                [
                    ("c_fc", Linear(
                        d_model,
                        mlp_width,
                        param_init_type=param_init_type,
                        compute_dtype=compute_dtype,
                    )),
                    ("gelu", QuickGELU() if self.use_clip else nn.GELU(approximate=False)),
                    ("c_proj", Linear(
                        mlp_width,
                        d_model,
                        param_init_type=param_init_type,
                        compute_dtype=compute_dtype,
                    )),
                ]
            )
        )
        self.ln_2 = LayerNorm((d_model,), eps=1e-6, param_init_type=ln_param_init_type)
        self.attn_mask = Parameter(attn_mask) if attn_mask is not None else None

        self.mlp_lora = []
        if not self.use_clip:
            for _ in range(lora_repeat_num):
                self.mlp_lora.append(LoraAdapter(
                    d_model=d_model,
                    out_feat=d_model,
                    r=32,
                    param_init_type=param_init_type,
                    compute_dtype=compute_dtype,
                ))
            self.mlp_lora = nn.CellList(self.mlp_lora)
        self.layernorm_compute_type = layernorm_compute_type

    def construct(self, x: Tensor, idx=None):
        residual0 = x
        x_type = x.dtype
        x = self.ln_1(x.to(self.layernorm_compute_type)).to(x_type)
        if self.use_clip:
            x = residual0 + self.attn(x)
        else:
            x = residual0 + self.attn(x, idx)
        residual1 = x
        x = self.ln_2(x.to(self.layernorm_compute_type)).to(x_type)
        x = residual1 + self.mlp(x)

        if not self.use_clip and idx is not None:
            x += self.mlp_lora[idx](residual1)
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
        compute_dtype=ms.float32,
        layernorm_compute_type=ms.float32,
        mlp_ratio=4.0,
        model_type="clip",
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.CellList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    attn_mask,
                    param_init_type=param_init_type,
                    ln_param_init_type=ln_param_init_type,
                    mlp_ratio=mlp_ratio,
                    model_type=model_type,
                    compute_dtype=compute_dtype,
                    layernorm_compute_type=layernorm_compute_type,
                )
                for _ in range(layers)
            ]
        )

    def construct(self, x: Tensor, idx=None):
        encoder_states = ()
        hidden_state = x
        hidden_state_list = list()
        for i, block in enumerate(self.resblocks):
            encoder_states += (hidden_state,)
            dtype = hidden_state.dtype
            if i > 20:  # After the 20th layer, the error of FP16 becomes unacceptable.
                hidden_state = hidden_state.to(ms.float32)
            hidden_state = block(hidden_state, idx)
            if i > 20:
                hidden_state = hidden_state.to(dtype)
            hidden_state_list.append(hidden_state)
        encoder_states += (hidden_state,)
        return encoder_states


class Resampler(nn.Cell):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
        self,
        grid_size,
        embed_dim,
        num_heads,
        positional_embedding_size=1024,
        kv_dim=None,
        param_init_type=ms.float32,
        ln_param_init_type=ms.float32,
        compute_dtype=ms.float32,
        layernorm_compute_type=ms.float32,
    ):
        super().__init__()
        self.num_queries = grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layernorm_compute_type = layernorm_compute_type

        self.pos_embed = Parameter(initializer("zeros", (positional_embedding_size, embed_dim), param_init_type))
        self.pos_embed_unsqueeze = Parameter(initializer("zeros", (256, embed_dim), param_init_type))

        self.query = Parameter(initializer("zeros", (self.num_queries, embed_dim), param_init_type))

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = Linear(
                kv_dim,
                embed_dim,
                has_bias=False,
                param_init_type=param_init_type,
                compute_dtype=compute_dtype,
            )
        else:
            self.kv_proj = None

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dtype=param_init_type)
        self.ln_q = LayerNorm(embed_dim, eps=1e-6, param_init_type=ln_param_init_type)
        self.ln_kv = LayerNorm(embed_dim, eps=1e-6, param_init_type=ln_param_init_type)

    def construct(self, x, attn_mask=None):
        pos_embed = self.pos_embed

        if self.kv_proj is not None:
            x = self.kv_proj(x)
        x = self.ln_kv(x.to(self.layernorm_compute_type)).to(x.dtype).permute(1, 0, 2)

        n = x.shape[1]
        q = self.ln_q(self.query.to(self.layernorm_compute_type)).to(self.query.dtype)
        out = self.attn(
            self._repeat(q, n) + self.pos_embed_unsqueeze.unsqueeze(1),
            x + pos_embed.unsqueeze(1),
            x,
            attn_mask=attn_mask)[0]
        return out.permute(1, 0, 2)

    @staticmethod
    def _repeat(query, n: int):
        return query.unsqueeze(1).tile((1, n, 1))


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
        compute_dtype=ms.float32,
        layernorm_compute_type=ms.float32,
        positional_embedding_size=None,
        mlp_ratio=4.0,
        model_type="clip",
    ):
        super().__init__()
        assert model_type in ("clip", "open_clip")
        self.use_clip = model_type == "clip"
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.vision_select_layer = vision_select_layer
        self.layernorm_compute_type = layernorm_compute_type

        scale = width**-0.5
        if positional_embedding_size is None:
            positional_embedding_size = (input_resolution // patch_size) ** 2 + 1
        self.positional_embedding = Parameter(
            (scale * ops.randn((positional_embedding_size, width))).astype(param_init_type)
        )
        self.ln_pre = LayerNorm((width,), eps=1e-5, param_init_type=ln_param_init_type)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            param_init_type=param_init_type,
            ln_param_init_type=ln_param_init_type,
            compute_dtype=compute_dtype,
            layernorm_compute_type=layernorm_compute_type,
            mlp_ratio=mlp_ratio,
            model_type=model_type,
        )
        if self.use_clip:
            self.class_embedding = Parameter((scale * ops.randn(width)).astype(param_init_type))
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
        else:
            self.attn_pool = Resampler(
                grid_size=16,
                embed_dim=output_dim,
                num_heads=output_dim // 128,
                positional_embedding_size=positional_embedding_size,
                kv_dim=width,
                param_init_type=param_init_type,
                ln_param_init_type=ln_param_init_type,
                compute_dtype=compute_dtype,
                layernorm_compute_type=layernorm_compute_type,
            )
            self.ln_post = LayerNorm((output_dim,), param_init_type=ln_param_init_type)
            self.proj = Parameter(initializer("normal", (output_dim, output_dim), param_init_type))
            self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=width,
                kernel_size=patch_size,
                stride=patch_size,
                has_bias=False,
                dtype=param_init_type,
            )

    def construct(self, x: Tensor, idx=None):
        if self.use_clip:
            x = self._clip_construct(x)
        else:
            x = self._open_clip_construct(x, idx)
        return x

    def _open_clip_construct(self, x, idx=None):
        # to patches
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = x + self.positional_embedding

        x = self.ln_pre(x.to(self.layernorm_compute_type)).to(x.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, idx=idx)[-1]
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.attn_pool(x)
        x = self.ln_post(x.to(self.layernorm_compute_type)).to(x.dtype)
        x = ops.matmul(x, self.proj)
        return x

    def _clip_construct(self, x: Tensor):
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
