import sys

import pytest

py_infer_path = "deploy/py_infer"
sys.path.insert(0, py_infer_path)

from src.data_process import build_postprocess

configs_list = [
    "configs/det/dbnet/db_r50_icdar15.yaml",
    "configs/rec/crnn/crnn_icdar15.yaml",
    "deploy/py_infer/src/configs/det/ppocr/det_r50_vd_sast_icdar15.yaml",
    "deploy/py_infer/src/configs/rec/mmocr/nrtr_resnet31-1by8-1by4_6e_st_mj.yaml",
]


@pytest.mark.parametrize("config_file", configs_list)
def test_build_postprocess(config_file):
    build_postprocess(config_file)
