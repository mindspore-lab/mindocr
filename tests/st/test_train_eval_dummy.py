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

def _create_combs():
    combs = set()
    task, val_while_train, gradient_accumulation_steps, clip_grad, grouping_strategy = 'rec', False, 1, False, None
    for task in ["det", "rec"]:
        for val_while_train in [False, True]:
            combs.add((task, val_while_train, gradient_accumulation_steps, clip_grad, grouping_strategy))

    val_while_train = False
    for task in ["det", "rec"]:
        for gradient_accumulation_steps in [1]: #[1, 2]:
            for clip_grad in [False, True]:
                combs.add((task, val_while_train, gradient_accumulation_steps, clip_grad, grouping_strategy))

    task, val_while_train, gradient_accumulation_steps, clip_grad, grouping_strategy = 'rec', False, 1, False, None
    for grouping_strategy in [None, 'filter_norm_and_bias']:
        for gradient_accumulation_steps in [1]: #[1, 2]:
            combs.add((task, val_while_train, gradient_accumulation_steps, clip_grad, grouping_strategy))
    print(combs)
    return list(combs)


# reduce combinations, only test val_while_train
'''
@pytest.mark.parametrize("task", ["det", "rec"])
@pytest.mark.parametrize("val_while_train", [False, True])
@pytest.mark.parametrize("gradient_accumulation_steps", [1, 2])
@pytest.mark.parametrize("clip_grad", [False, True])
@pytest.mark.parametrize("grouping_strategy", [None]) #[None, 'filter_norm_and_bias'])
'''
@pytest.mark.parametrize("task, val_while_train, gradient_accumulation_steps, clip_grad, grouping_strategy", _create_combs())
def test_train_eval(task, val_while_train, gradient_accumulation_steps, clip_grad, grouping_strategy):

    # prepare dummy images
    data_dir = gen_dummpy_data(task)

     # modify ocr predefined yaml for minimum test
    if task == 'det':
        config_fp = 'configs/det/dbnet/db_r50_icdar15.yaml'
    elif task=='rec':
        #config_fp = 'configs/rec/vgg7_bilstm_ctc.yaml' # TODO: change on lmdb datasset
        config_fp = 'configs/rec/crnn/crnn_icdar15.yaml'

    dummpy_config_fp = update_config_for_CI(config_fp,
                                            task,
                                            val_while_train=val_while_train,
                                            gradient_accumulation_steps=gradient_accumulation_steps,
                                            clip_grad=clip_grad,
                                            grouping_strategy=grouping_strategy,
                                            )

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
    #test_train_eval('rec', True, 1, True, 'filter_norm_and_bias')
    #test_train_eval('det', True, 1, False, None)
    test_train_eval('rec', True, 1, True, 'filter_norm_and_bias')
    #test_train_eval('rec', True, 2, False, None)
