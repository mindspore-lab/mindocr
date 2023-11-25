import argparse
import json
import os

import numpy as np

from mindspore import Parameter, save_checkpoint


def convert_hepler(input_path: str, json_path: str, output_path: str):
    if os.path.exists(input_path) is not True:
        raise ValueError("The pp_ckpt_path doesn't exist.")
    if os.path.exists(json_path) is not True:
        raise ValueError("The json path doesn't exist.")

    if output_path.endswith(".ckpt") is True:
        output_dir, output_filename = os.path.split(os.path.abspath(output_path))
    else:
        output_dir, output_filename = os.path.abspath(output_path), "from_paddle.ckpt"
    if os.path.exists(output_dir) is not True:
        os.mkdir(output_dir)
    real_output_path = os.path.join(output_dir, output_filename)

    pp_ckpt = np.load(input_path, allow_pickle=True)
    ms_ckpt = list()

    with open(json_path, "r") as json_file:
        helper_json = json.load(json_file)
    convert_map = helper_json["convert_map"]
    transpose_map = helper_json["transpose_map"]
    for pp_name, ms_name in convert_map.items():
        np_param = pp_ckpt[pp_name]
        if transpose_map[pp_name] is True:
            np_param = np_param.transpose(1, 0)
        ms_name = convert_map[pp_name]
        ms_ckpt.append({"name": ms_name, "data": Parameter(np_param)})

    save_checkpoint(ms_ckpt, real_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the path of paddle pdparams, and convert it to mindspore ckpt.")
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        default="ser_vi_layoutxlm_xfund_pretrained/best_accuracy/model_state.pdparams",
        help="The input path of the paddle pdparams.",
    )
    parser.add_argument(
        "-j",
        "--json_path",
        type=str,
        default="mindocr/model/backbone/layoutxlm/ser_layoutxlm_helper.json",
        help="The path of the json.",
    )
    parser.add_argument(
        "-o", "--output_path", type=str, default="from_paddle.ckpt", help="The output path of the mindspore ckpt."
    )
    args = parser.parse_args()

    convert_hepler(input_path=args.input_path, json_path=args.json_path, output_path=args.output_path)
