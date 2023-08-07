# convert pytorch model to mindspore
import json
import os

from transformers import LayoutLMv2Model

import mindspore as ms


def set_proxy():
    proxy_addr = "http://127.0.0.1:7078"  # your proxy addr
    os.environ['http_proxy'] = proxy_addr
    os.environ['https_proxy'] = proxy_addr


def unset_proxy():
    os.environ.pop("http_proxy")
    os.environ.pop("https_proxy")


# load param_map json
with open("param_map.json", "r") as json_file:
    param_name_map = json.load(json_file)

# use proxy if you needed
set_proxy()

# load pytorch model
model = LayoutLMv2Model.from_pretrained("microsoft/layoutxlm-base")
params_dict = model.state_dict()

# conversion
ms_params = []
for name, value in params_dict.items():
    each_param = dict()
    each_param[param_name_map[name]] = ms.Tensor(value.numpy())
    ms_params.append(each_param)

ms.save_checkpoint(ms_params, "layoutxlm-base.ckpt")

unset_proxy()
