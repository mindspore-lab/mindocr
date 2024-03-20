"""Convert Vary Toy weight."""

import argparse

import torch

import mindspore as ms

ATTENTION_WEIGHT_NAME = "attn.c_attn.weight"
ATTENTION_BIAS_NAME = "attn.c_attn.bias"


def pt2ms(value: torch.Tensor, dtype) -> ms.Tensor:
    """
    convert torch.Tensor to ms.Tensor with specified dtype
    """
    if value.dtype == torch.bfloat16:
        np_value = value.to(torch.float32).numpy()
    else:
        np_value = value.detach().numpy()

    if dtype:
        return ms.Tensor(np_value, dtype=dtype)
    return ms.Tensor(np_value, dtype=ms.bfloat16) if value.dtype == torch.bfloat16 else ms.Tensor(np_value)


def _name_replace(name: str):
    # qwen
    name = name.replace(".h.", ".layers.")
    name = name.replace(".wte.weight", ".wte.embedding_weight")
    name = name.replace("attn.c_proj.", "attention.wo.")
    name = name.replace("ln_1.", "attention_norm.")
    name = name.replace("ln_2.", "ffn_norm.")
    name = name.replace("mlp.w1.", "feed_forward.w1.")
    name = name.replace("mlp.w2.", "feed_forward.w3.")
    name = name.replace("mlp.c_proj.", "feed_forward.w2.")

    # clip
    name = name.replace("vision_model.", "")
    name = name.replace("embeddings.", "")
    name = name.replace("patch_embedding.", "conv1.")
    name = name.replace("position_embedding.weight", "positional_embedding")
    name = name.replace("pre_layrnorm.weight", "ln_pre.gamma")
    name = name.replace("pre_layrnorm.bias", "ln_pre.beta")
    name = name.replace("encoder.layers", "transformer.resblocks")
    name = name.replace("layer_norm1.weight", "ln_1.gamma")
    name = name.replace("layer_norm1.bias", "ln_1.beta")
    name = name.replace("fc1", "c_fc")
    name = name.replace("fc2", "c_proj")
    name = name.replace("layer_norm2.weight", "ln_2.gamma")
    name = name.replace("layer_norm2.bias", "ln_2.beta")
    name = name.replace("self_attn", "attn")
    name = name.replace("post_layernorm", "vision_model.post_layernorm")

    # sam
    name = name.replace("norm1.weight", "norm1.gamma")
    name = name.replace("norm1.bias", "norm1.beta")
    name = name.replace("norm2.weight", "norm2.gamma")
    name = name.replace("norm2.bias", "norm2.beta")
    return name


def convert_attention_weight(name, value, ckpt_weights):
    split_value = ms.numpy.array_split(value, 3)
    attention_weight_names = ["attention.wq.weight", "attention.wk.weight", "attention.wv.weight"]

    for index in range(len(split_value)):
        cur_name = name.replace(ATTENTION_WEIGHT_NAME, attention_weight_names[index])
        ckpt_weights.append({"name": cur_name, "data": ms.Tensor(split_value[index])})


def convert_attention_bias(name, value, ckpt_weights):
    split_value = ms.numpy.array_split(value, 3)
    attention_bias_names = ["attention.wq.bias", "attention.wk.bias", "attention.wv.bias"]

    for index in range(len(split_value)):
        cur_name = name.replace(ATTENTION_BIAS_NAME, attention_bias_names[index])
        ckpt_weights.append({"name": cur_name, "data": ms.Tensor(split_value[index])})


def convert_pt_to_ms(torch_ckpt_path, output_path, dtype=ms.float16):
    state_dict = torch.load(torch_ckpt_path, map_location="cpu")
    ckpt_weights = []
    for k, v in state_dict.items():
        value = pt2ms(v, dtype)

        msname = _name_replace(k)

        if msname != k:
            print("name:  %s->%s" % (k, msname))

        if ATTENTION_WEIGHT_NAME in msname:
            convert_attention_weight(msname, value, ckpt_weights)
            continue

        if ATTENTION_BIAS_NAME in msname:
            convert_attention_bias(msname, value, ckpt_weights)
            continue

        ckpt_weights.append({"name": msname, "data": value})

    print("Saving converted weights to %s..." % output_path)
    ms.save_checkpoint(ckpt_weights, output_path)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vary convert script")
    parser.add_argument("--torch_ckpt_path", required=True, help="The torch checkpoint path.")
    parser.add_argument("--mindspore_ckpt_path", default="./vary_toy.ckpt", help="The output checkpoint path.")

    args = parser.parse_args()

    convert_pt_to_ms(args.torch_ckpt_path, args.mindspore_ckpt_path, ms.float16)
