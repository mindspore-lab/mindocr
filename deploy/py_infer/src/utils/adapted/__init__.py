import os

from .mindocr_models import MINDOCR_CONFIG_PATH, MINDOCR_MODELS
from .mmocr_models import MMOCR_CONFIG_PATH, MMOCR_MODELS
from .paddleocr_models import PADDLEOCR_CONFIG_PATH, PADDLEOCR_MODELS

__all__ = ["get_config_by_name_for_model"]


def get_config_by_name_for_model(name: str):
    if name in MINDOCR_MODELS:
        return os.path.abspath(os.path.join(MINDOCR_CONFIG_PATH, MINDOCR_MODELS[name]))
    elif name in PADDLEOCR_MODELS:
        return os.path.abspath(os.path.join(PADDLEOCR_CONFIG_PATH, PADDLEOCR_MODELS[name]))
    elif name in MMOCR_MODELS:
        return os.path.abspath(os.path.join(MMOCR_CONFIG_PATH, MMOCR_MODELS[name]))
    else:
        raise ValueError(f"The model name {name} is not supported, please check the model name.")
