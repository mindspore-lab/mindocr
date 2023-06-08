import sys

sys.path.append(".")
import pytest
import yaml
from addict import Dict

from mindocr import build_loss


@pytest.mark.parametrize("task", ["det", "rec"])
def test_build_loss(task):
    if task == "det":
        config_fp = "configs/det/dbnet/db_r50_icdar15.yaml"
    elif task == "rec":
        config_fp = "configs/rec/crnn/crnn_icdar15.yaml"

    with open(config_fp) as fp:
        cfg = yaml.safe_load(fp)
    cfg = Dict(cfg)

    build_loss(cfg.loss.pop("name"), **cfg["loss"])
