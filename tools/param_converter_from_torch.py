import argparse
import json
import os

import torch

from mindspore import Parameter, save_checkpoint


def convert_hepler(input_path: str, json_path: str, output_path: str):
    if os.path.exists(input_path) is not True:
        raise ValueError("The torch_ckpt_path doesn't exist.")
    if os.path.exists(json_path) is not True:
        raise ValueError("The json path doesn't exist.")

    output_dir, output_filename = os.path.split(os.path.abspath(output_path))

    if os.path.exists(output_dir) is not True:
        os.mkdir(output_dir)
    real_output_path = os.path.join(output_dir, output_filename)

    pt_ckpt = torch.load(input_path, map_location=torch.device("cpu"), weights_only=False)["model"]
    ms_ckpt = list()

    with open(json_path, "r") as json_file:
        helper_json = json.load(json_file)
    convert_map = helper_json["convert_map"]
    for pt_name, ms_name in convert_map.items():
        np_param = pt_ckpt[pt_name].detach().numpy()
        ms_name = convert_map[pt_name]
        ms_ckpt.append({"name": ms_name, "data": Parameter(np_param)})

    save_checkpoint(ms_ckpt, real_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the path of paddle pdparams, and convert it to mindspore ckpt.")
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        default="model_final.pt",
        help="The input path of the paddle pdparams.",
    )
    parser.add_argument(
        "-j",
        "--json_path",
        type=str,
        default="configs/layout/layoutlmv3/layoutlmv3_publaynet_param_map.json",
        help="The path of the json.",
    )
    parser.add_argument(
        "-o", "--output_path", type=str, default="from_torch.ckpt", help="The output path of the mindspore ckpt."
    )
    args = parser.parse_args()

    convert_hepler(input_path=args.input_path, json_path=args.json_path, output_path=args.output_path)
