import os

import yaml

from .mindocr_models import MINDOCR_CONFIG_PATH, MINDOCR_MODELS
from .mmocr_models import MMOCR_CONFIG_PATH, MMOCR_MODELS
from .paddleocr_models import PADDLEOCR_CONFIG_PATH, PADDLEOCR_MODELS

__all__ = ["get_config_by_name_for_model"]


def get_config_by_name_for_model(model_name_or_config: str):
    if os.path.isfile(model_name_or_config):
        filename = model_name_or_config
    elif model_name_or_config in MINDOCR_MODELS:
        filename = os.path.abspath(os.path.join(MINDOCR_CONFIG_PATH, MINDOCR_MODELS[model_name_or_config]))
    elif model_name_or_config in PADDLEOCR_MODELS:
        filename = os.path.abspath(os.path.join(PADDLEOCR_CONFIG_PATH, PADDLEOCR_MODELS[model_name_or_config]))
    elif model_name_or_config in MMOCR_MODELS:
        filename = os.path.abspath(os.path.join(MMOCR_CONFIG_PATH, MMOCR_MODELS[model_name_or_config]))
    else:
        raise ValueError(
            f"The {model_name_or_config} must be a model name or YAML config file path, "
            "please check whether the file exists, or whether model name is in the supported models list."
        )

    with open(filename) as fp:
        cfg = yaml.safe_load(fp)

    try:
        cfg["eval"]["dataset"]["transform_pipeline"]
        cfg["postprocess"]
    except KeyError:
        preprocess_desc = "{eval: {dataset: {transform_pipeline: ...}}}"
        postprocess_desc = "{postprocess: ...}"
        raise ValueError(
            f"The YAML config file {filename} must contain preprocess pipeline key {preprocess_desc} "
            f"and postprocess key {postprocess_desc}."
        )

    return filename
