import numpy as np
import pytest
import sys

py_infer_path = "deploy/py_infer"
sys.path.insert(0, py_infer_path)

from src.data_process import build_preprocess


@pytest.mark.parametrize('task', ['det', 'rec'])
def test_transforms_pipeline(task):
    if task == 'det':
        config_fp = 'configs/det/dbnet/db_r50_icdar15.yaml'
        image_shape = (720, 1280, 3)
        target_size = (736, 1280)
    elif task == 'rec':
        config_fp = 'configs/rec/crnn/crnn_icdar15.yaml'
        image_shape = (34, 114, 3)
        target_size = (32, 100)

    preprocess_ops = build_preprocess(config_fp)
    image = np.random.randint(0, 255, size=image_shape).astype(np.float32)
    data = preprocess_ops(image, target_size=target_size)

    assert data['image'].shape == (1, 3) + target_size

    if task == 'det':
        assert data['shape_list'].shape == (1, 4)
