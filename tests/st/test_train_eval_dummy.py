'''
Using a dummy images and ocr labels to test the train and eval pipeline.
Expected results: run pass, but loss may not decrease.

Example:
    pytest tests/st/test_train_eval_dummy.py

'''

import os
import subprocess
import sys
import glob
import yaml
import pytest

sys.path.append(".")

from tests.ut._common import gen_dummpy_data, update_config_for_CI
from mindocr.models.backbones.mindcv_models.download import DownLoad


@pytest.mark.parametrize("task", ["det", "rec"])
@pytest.mark.parametrize("val_while_train", [False, True])
def test_train_eval(task, val_while_train):

    # prepare dummy images
    data_dir = gen_dummpy_data(task)

     # modify ocr predefined yaml for minimum test
    if task == 'det':
        config_fp = 'configs/det/dbnet/db_r50_icdar15.yaml'
    elif task=='rec':
        #config_fp = 'configs/rec/vgg7_bilstm_ctc.yaml' # TODO: change on lmdb datasset
        config_fp = 'configs/rec/crnn/crnn_icdar15.yaml'

    dummpy_config_fp = update_config_for_CI(config_fp, task)

    #dummpy_config_fp = 'tests/st/rec_crnn_test.yaml'
    # ---------------- test running train.py using the toy data ---------

    cmd = (
        f"python tools/train.py --config {dummpy_config_fp}"
    )

    print(f"Running command: \n{cmd}")
    ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    assert ret == 0, "Training fails"

    # --------- Test running validate.py using the trained model ------------- #
    # begin_ckpt = os.path.join(ckpt_dir, f'{model}-1_1.ckpt')
    cmd = (
        f"python tools/eval.py --config {dummpy_config_fp}"
    )
    # ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    print(f"Running command: \n{cmd}")
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    out, err = p.communicate()
    # assert ret==0, 'Validation fails'
    print(out)

    p.kill()


if __name__ == '__main__':
    #test_train_eval('det', True)
    test_train_eval('rec', True)
