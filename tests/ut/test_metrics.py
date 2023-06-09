import sys

sys.path.append(".")
import numpy as np
import pytest
import yaml
from addict import Dict

import mindspore as ms

from mindocr import build_metric
from mindocr.metrics.det_metrics import DetMetric
from mindocr.metrics.rec_metrics import RecMetric


@pytest.mark.parametrize("task", ["det", "rec"])
@pytest.mark.parametrize("device_num", [1, 8])
def test_build_metric(task, device_num):
    if task == "det":
        config_fp = "configs/det/dbnet/db_r50_icdar15.yaml"
    elif task == "rec":
        config_fp = "configs/rec/crnn/crnn_icdar15.yaml"

    with open(config_fp) as fp:
        cfg = yaml.safe_load(fp)
    cfg = Dict(cfg)

    build_metric(cfg.metric, device_num=device_num)


def test_det_metric():
    pred_polys = [
        [
            [[0, 0], [0, 10], [10, 10], [10, 0]],
            [[10, 10], [10, 20], [20, 20], [20, 10]],
            [[20, 20], [20, 30], [30, 30], [30, 20]],
        ]
    ]
    pred_polys = np.array(pred_polys, dtype=np.float32)
    confs = np.array([[1.0, 0.8, 0.9]])
    preds = {"polys": pred_polys, "scores": confs}

    gt_polys = [
        [
            [[0, 0], [0, 9], [9, 9], [9, 0]],
            [[10, 10], [-10, -20], [-20, -20], [-20, -10]],
            [[20, 20], [20, 30], [30, 30], [30, 20]],
        ]
    ]
    gt_polys = ms.Tensor(np.array(gt_polys, dtype=np.float32))
    ignore_tags = ms.Tensor([[False, False, True]])
    gts = (gt_polys, ignore_tags)

    m = DetMetric()
    m.update(preds, gts)

    perf = m.eval()
    print(perf)

    # check correctness
    assert perf["recall"] == 0.5
    assert perf["precision"] == 0.5
    assert perf["f-score"] == 0.5


def test_rec_metric():
    gt = ["ba la la!    ", "ba       "]
    gt_len = [len("ba xla la!"), len("ba")]
    pred = ["baxlala", "ba"]

    m = RecMetric()
    m.update({"texts": pred}, (gt, gt_len))
    perf = m.eval()
    print(perf)

    # check correctness
    assert perf["acc"] == 0.5
    assert (perf["norm_edit_distance"] - 0.92857) < 1e-4


if __name__ == "__main__":
    test_det_metric()
    # test_rec_metric()
