import argparse
import logging
import os
import subprocess
import sys

import yaml
from addict import Dict

sys_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(sys_path)

from src import auto_scaling_process  # noqa

with open(os.path.abspath(os.path.join(sys_path, "configs/auto_scaling.yaml")), "r") as fp:
    config = yaml.safe_load(fp)
config = Dict(config)

logging.getLogger().setLevel(logging.INFO)


def get_safe_name(path):
    """Remove ending path separators before retrieving the basename.

    e.g. /xxx/ -> /xxx
    """
    return os.path.basename(os.path.abspath(path))


def custom_islink(path):
    """Remove ending path separators before checking soft links.

    e.g. /xxx/ -> /xxx
    """
    return os.path.islink(os.path.abspath(path))


def check_path_valid(path):
    name = get_safe_name(path)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Error! {name} must exist!")
    if custom_islink(path):
        raise ValueError(f"Error! {name} cannot be a soft link!")
    if not os.access(path, mode=os.R_OK):
        raise ValueError(f"Error! Please check if {name} is readable.")


def args_check(opts):
    if opts.dataset_path:
        check_path_valid(opts.dataset_path)
    if opts.model_path:
        check_path_valid(opts.model_path)
    if opts.input_shape:
        split_shape = opts.input_shape.split(",")
        if len(split_shape) != 4:
            raise ValueError("Error! Please check input_shape is correct.")
        if split_shape[1].strip() != "3":
            raise ValueError("Error! Channel must be 3.")


def parse_args():
    parser = argparse.ArgumentParser()
    # scaling related
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--backend",
        type=str.lower,
        required=False,
        default="lite",
        choices=["atc", "lite"],
    )
    parser.add_argument("--input_name", type=str, required=False, default="x")
    parser.add_argument("--input_shape", type=str, required=False, default="-1,3,-1,-1")
    parser.add_argument("--dataset_path", type=str, required=False)
    parser.add_argument("--output_path", type=str, required=False, default="output")

    # backend related
    parser.add_argument(
        "--soc_version",
        type=str,
        required=False,
        default="Ascend310P3",
        choices=["Ascend310P3", "Ascend310"],
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    _args = parse_args()
    args_check(_args)

    subp_list = auto_scaling_process(_args, config, sys_path)
    for subp in subp_list:
        try:
            subp.wait(3600)
        except subprocess.TimeoutExpired:
            logging.error("Error! Conversion time more than 1 hour!")
            sys.exit(-1)
        finally:
            subp.kill()

    logging.info("Converter finish!")
