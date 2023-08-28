import os
import sys
import threading

import yaml

import mindspore as ms

from ...utils import suppress_stderr
from . import adapted_postprocess

mindocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
sys.path.insert(0, mindocr_path)

from mindocr.postprocess import builder as mindocr_postprocess  # noqa

__all__ = ["build_postprocess"]


class Postprocessor:
    def __init__(self, tasks, **kwargs):
        ops, params = list(tasks.items())[0]
        params.update(**kwargs)
        self._ops_func = ops(**params)

        # if check device failed, set device_target="CPU"
        if get_device_status() == 1:
            # FIXME: set_context may be invalid sometimes, it's best to patch to XXXPostprocess.__init__
            ms.set_context(device_target="CPU")

    def __call__(self, *args, **kwargs):
        return self._ops_func(*args, **kwargs)


def get_device_status():
    """
    If device is Ascend310/310P, ops.xxx may be unavailable, then return status=1; otherwise status=0.
    """
    if ms.get_context("device_target") != "Ascend":
        return 0

    status = 1

    def _get_status():
        nonlocal status
        status = ms.Tensor([0])[0:].asnumpy()[0]
        return status

    with suppress_stderr():
        test_thread = threading.Thread(target=_get_status)
        test_thread.start()
        test_thread.join()

    return status


def parse_postprocess_from_yaml(config_path):
    with open(config_path) as fp:
        cfg = yaml.safe_load(fp)

    postprocess_params = cfg["postprocess"]

    postprocess_name = postprocess_params["name"]

    # Mark 'from_ppocr' or 'from_mmocr' for some conflict postprocess
    must_load_adapted = False
    marked_list = ["from_ppocr", "from_mmocr"]
    for marked in marked_list:
        must_load_adapted = must_load_adapted or postprocess_params.get(marked, False)
        postprocess_params.pop(marked, None)

    # mindocr_postprocess > adapted_postprocess
    if (not must_load_adapted) and hasattr(mindocr_postprocess, postprocess_name):
        postprocess = getattr(mindocr_postprocess, postprocess_name)
    elif hasattr(adapted_postprocess, postprocess_name):
        postprocess = getattr(adapted_postprocess, postprocess_name)
    else:
        raise ValueError(f"The postprocess '{postprocess_name}' is not supported yet.")

    postprocess_params.pop("name")
    return {postprocess: postprocess_params}


def build_postprocess(config_path, **kwargs):
    tasks = parse_postprocess_from_yaml(config_path)
    processor = Postprocessor(tasks, **kwargs)
    return processor
