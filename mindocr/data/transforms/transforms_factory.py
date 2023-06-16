"""
Create and run transformations from a config or predefined transformation pipeline
"""
from typing import Dict, List

import numpy as np

from .det_east_transforms import *
from .det_fce_transforms import *
from .det_transforms import *
from .general_transforms import *
from .rec_transforms import *
from .svtr_transform import *

__all__ = ["create_transforms", "run_transforms", "transforms_dbnet_icdar15"]


# TODO: use class with __call__, to perform transformation
def create_transforms(transform_pipeline: List, global_config: Dict = None):
    """
    Create a squence of callable transforms.

    Args:
        transform_pipeline (List): list of callable instances or dicts where each key is a transformation class name,
            and its value are the args.
            e.g. [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}}]
                 [DecodeImage(img_mode='BGR')]

    Returns:
        list of data transformation functions
    """
    assert isinstance(
        transform_pipeline, list
    ), f"transform_pipeline config should be a list, but {type(transform_pipeline)} detected"

    transforms = []
    for transform_config in transform_pipeline:
        if isinstance(transform_config, dict):
            assert len(transform_config) == 1, "yaml format error in transforms"
            trans_name = list(transform_config.keys())[0]
            param = {} if transform_config[trans_name] is None else transform_config[trans_name]
            if global_config is not None:
                param.update(global_config)
            # TODO: assert undefined transform class

            transform = eval(trans_name)(**param)
            transforms.append(transform)
        elif callable(transform_config):
            transforms.append(transform_config)
        else:
            raise TypeError("transform_config must be a dict or a callable instance")
        # print(global_config)
    return transforms


def run_transforms(data, transforms=None, verbose=False):
    if transforms is None:
        transforms = []
    for i, transform in enumerate(transforms):
        if verbose:
            print(f"Trans {i}: ", transform)
            print("\tInput: ", {k: data[k].shape for k in data if isinstance(data[k], np.ndarray)})
        data = transform(data)
        if verbose:
            print("\tOutput: ", {k: data[k].shape for k in data if isinstance(data[k], np.ndarray)})

        if data is None:
            raise RuntimeError("Empty result is returned from transform `{transform}`")
    return data


# ---------------------- Predefined transform pipeline ------------------------------------
def transforms_dbnet_icdar15(phase="train"):
    """
    Get pre-defined transform config for dbnet on icdar15 dataset.
    Args:
        phase: train, eval, infer
    Returns:
        list of dict for data transformation pipeline, which can be convert to functions by 'create_transforms'
    """
    if phase == "train":
        pipeline = [
            {"DecodeImage": {"img_mode": "RGB", "to_float32": False}},
            {"DetLabelEncode": None},
            {"RandomScale": {"scale_range": [1.022, 3.0]}},
            {"IaaAugment": {"Affine": {"rotate": [-10, 10]}, "Fliplr": {"p": 0.5}}},
            {"RandomCropWithBBox": {"max_tries": 100, "min_crop_ratio": 0.1, "crop_size": (640, 640)}},
            {"ShrinkBinaryMap": {"min_text_size": 8, "shrink_ratio": 0.4}},
            {
                "BorderMap": {
                    "shrink_ratio": 0.4,
                    "thresh_min": 0.3,
                    "thresh_max": 0.7,
                }
            },
            {"RandomColorAdjust": {"brightness": 32.0 / 255, "saturation": 0.5}},
            {
                "NormalizeImage": {
                    "bgr_to_rgb": False,
                    "is_hwc": True,
                    "mean": [123.675, 116.28, 103.53],
                    "std": [58.395, 57.12, 57.375],
                }
            },
            {"ToCHWImage": None},
        ]

    elif phase == "eval":
        pipeline = [
            {"DecodeImage": {"img_mode": "RGB", "to_float32": False}},
            {"DetLabelEncode": None},
            {"GridResize": {"factor": 32}},
            {"ScalePadImage": {"target_size": [736, 1280]}},
            {
                "NormalizeImage": {
                    "bgr_to_rgb": False,
                    "is_hwc": True,
                    "mean": [123.675, 116.28, 103.53],
                    "std": [58.395, 57.12, 57.375],
                }
            },
            {"ToCHWImage": None},
        ]
    else:
        pipeline = [
            {"DecodeImage": {"img_mode": "RGB", "to_float32": False}},
            {"GridResize": {"factor": 32}},
            {"ScalePadImage": {"target_size": [736, 1280]}},
            {
                "NormalizeImage": {
                    "bgr_to_rgb": False,
                    "is_hwc": True,
                    "mean": [123.675, 116.28, 103.53],
                    "std": [58.395, 57.12, 57.375],
                }
            },
            {"ToCHWImage": None},
        ]
    return pipeline
