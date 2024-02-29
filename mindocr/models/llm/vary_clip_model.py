from collections import OrderedDict
import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops, load_checkpoint, load_param_into_net


class QuickGELU(nn.Cell):
    def construct(self, x: Tensor):
        return x * ops.sigmoid(1.702 * x)
 

class CLIPAttention(nn.Cell):
    """Multi-head attention module for CLIP"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.k_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.batch_matmul = ops.BatchMatMul()
        self.batch_matmul_q_k = ops.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax()

    def construct(self, x):
        bsz, tgt_len, embed_dim = x.shape
        query_states = self.transpose(self.reshape(self.q_proj(x) * self.scale, (bsz, tgt_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        key_states = self.transpose(self.reshape(self.k_proj(x), (bsz, tgt_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        value_states = self.transpose(self.reshape(self.v_proj(x), (bsz, tgt_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

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
    def __init__(self, d_model: int, n_head: int, attn_mask: Tensor = None):
        super().__init__()

        self.attn = CLIPAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.mlp = nn.SequentialCell(
            OrderedDict(
                [
                    ("c_fc", nn.Dense(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Dense(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm([d_model], epsilon=1e-5)
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
    def __init__(self, width: int, layers: int, heads: int, attn_mask: Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.CellList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])


    def construct(self, x: Tensor):
        encoder_states = ()
        hidden_state = x
        for block in self.resblocks:
            encoder_states += (hidden_state, )
            hidden_state = block(hidden_state)
        encoder_states += (hidden_state,)
        return encoder_states


class VisionTransformer(nn.Cell):
    """CLIP module for Vary system"""
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, vision_select_layer: int):
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
        )

        scale = width**-0.5
        self.class_embedding = Parameter(scale * ops.randn(width))
        self.positional_embedding = Parameter(scale * ops.randn(((input_resolution // patch_size) ** 2 + 1, width)))
        self.ln_pre = nn.LayerNorm([width], epsilon=1e-5)
        self.transformer = Transformer(width, layers, heads)

    def construct(self, x: Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape((x.shape[0], x.shape[1], -1))  # shape = [*, width, grid ** 2]
        x = x.permute((0, 2, 1))  # shape = [*, grid ** 2, width]
        x = ops.cat(
            [self.class_embedding.to(x.dtype) + ops.zeros((x.shape[0], 1, x.shape[-1]), dtype=x.dtype), x], axis=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)  # torch version: CLIPVisionEmbeddings
        x_type = x.dtype
        x = self.ln_pre(x.to(ms.float32)).to(x_type)  # modeling_clip.py L842
        x = self.transformer(x)  # modeling_clip.py L844, error 1e-3
        x = x[self.vision_select_layer][:, 1:]
        return x


def convert_weights(model: nn.Cell):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(layer):
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Dense)):
            layer.weight.to(ms.float16)
            if layer.bias is not None:
                layer.bias.to(ms.float16)

        if isinstance(layer, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                param = getattr(layer, attr)
                if param is not None:
                    param.to(ms.float16)

        for name in ["text_projection", "proj"]:
            if hasattr(layer, name):
                attr = getattr(layer, name)
                if attr is not None:
                    attr.to(ms.float16)

    model.apply(_convert_weights_to_fp16)


def build_model(ckpt_dict: dict):
    """construct the CLIP module and load ckpt"""
    vision_width = ckpt_dict["visual.conv1.weight"].shape[0]
    vision_layers = len(
        [k for k in ckpt_dict.keys() if k.startswith("visual.") and k.endswith(".attn.k_proj.weight")]
    )
    vision_patch_size = ckpt_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((ckpt_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    out_width = ckpt_dict["visual.transformer.resblocks." + str(vision_layers-1) + ".ln_2.gamma"].shape[0]
 
    model = VisionTransformer(
        input_resolution=image_resolution,  # image_size in transformers
        patch_size=vision_patch_size,  # patch_size in transformers
        width=vision_width,  # hidden_size
        layers=vision_layers,  # num_hidden_layers
        heads=grid_size,  # num_attention_heads
        output_dim=out_width,  # projection_dim in transformers, default: 1024
        vision_select_layer = -2,
    )
 
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in ckpt_dict:
            del ckpt_dict[key]
 
    load_param_into_net(model, ckpt_dict)
    convert_weights(model)
    return model.set_train(False)


if __name__ == "__main__":
    """
    realize following torch code in vary_qwen_vary.py using MindSpore:

    with torch.set_grad_enabled(False):
        image_forward_out = vision_tower(image[0], output_hidden_states=True)
        select_hidden_state = image_forward_out.hidden_states[vision_select_layer]
        image_feature = select_hidden_state[:, 1:]  # 256*1024
    """
    ms.set_context(device_target="GPU", mode=ms.PYNATIVE_MODE)
    img = np.load("/data0/perf/git/mindocr/tests/ut/clip_img.npy").astype(np.float16)
    img_ts = Tensor.from_numpy(img)

    ckpt_dict = load_checkpoint("/data0/perf/code/wtc_ms/Vary/Vary-master/vit-large-patch14/pytorch_model.ckpt")
    model = build_model(ckpt_dict)
    model.to_float(ms.float16)
    image_feature = model(img_ts)
    print("test clip completed")
