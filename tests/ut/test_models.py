import subprocess
import sys

from _common import gen_dummpy_data, update_config_for_CI

sys.path.append(".")

import pytest

import mindocr
from mindocr.models import build_model

all_model_names = mindocr.list_models()
print("Registered models: ", all_model_names)

all_yamls = [
    "configs/det/dbnet/db_r50_icdar15.yaml",
    "configs/det/dbnet/db++_r50_icdar15.yaml",
    "configs/rec/crnn/crnn_resnet34.yaml",
    "configs/rec/master/master_resnet31.yaml",
    "configs/rec/rare/rare_resnet34.yaml",
    "configs/rec/svtr/svtr_tiny.yaml",
    "configs/rec/visionlan/visionlan_resnet45_LF_1.yaml",
    "configs/cls/mobilenetv3/cls_mv3.yaml",
    "configs/rec/abinet/abinet_resnet45_en.yaml",
]
print("All config yamls: ", all_yamls)


@pytest.mark.parametrize("model_name", all_model_names)
@pytest.mark.parametrize("pretrained", [True, False])
def test_model_by_name(model_name, pretrained):
    print(model_name)
    build_model(model_name, pretrained=pretrained)
    print("model created")


gen_dummpy_data(task="det")
gen_dummpy_data(task="rec")
gen_dummpy_data(task="cls")


@pytest.mark.parametrize("yaml_fp", all_yamls)
def test_model_by_yaml(yaml_fp):
    task = yaml_fp.split("/")[1]
    dummpy_config_fp = update_config_for_CI(yaml_fp, task, val_while_train=True)
    # ---------------- test running train.py using the toy data ---------

    cmd = f"python tools/train.py --config {dummpy_config_fp}"

    print(f"Running command: \n{cmd}")
    ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    assert ret == 0, "Training fails"


if __name__ == "__main__":
    print(all_model_names)
    test_model_by_yaml(all_yamls[0])
