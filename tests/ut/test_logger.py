import sys

sys.path.append(".")

import pytest
import yaml
from addict import Dict

from mindocr.utils.logger import set_logger


@pytest.mark.parametrize("task", ["det", "rec"])
def test_logger(task):
    if task == "det":
        config_fp = "configs/det/dbnet/db_r50_icdar15.yaml"
    elif task == "rec":
        config_fp = "configs/rec/crnn/crnn_icdar15.yaml"

    with open(config_fp) as fp:
        cfg = yaml.safe_load(fp)
    cfg = Dict(cfg)

    set_logger(name="mindocr", output_dir=cfg.train.ckpt_save_dir, rank=0)
