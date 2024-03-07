import sys
from functools import wraps

sys.path.append(".")

import yaml

from mindocr.models.utils.deepsolo.deepsolo_layers import *
from mindocr.models.utils.deepsolo.deformable_transformer import *
from mindocr.utils.seed import set_seed
from test_utils import *
import mindspore as ms

@test_op_wrapper
def test_position_embedding_2d(config):
    test_config = config["PositionEmbedding2D"]
    data_list = generate_data(test_config["data_config"])
    op = eval("PositionEmbedding2D")
    op_config = check_config(test_config["op_config"])
    # rsts = debug_op(op, op_config, data_list)
    rsts = run_op(op, op_config, data_list)
    success = check(rsts, test_config["targets"])
    return success
    
@test_op_wrapper
def test_ms_deform_attn_core(config):
    test_config = config["ms_deform_attn_core"]
    data_list = generate_data(test_config["data_config"])
    op = eval("ms_deform_attn_core")
    # rsts = debug_op(op, None, data_list)
    rsts = run_op(op, None, data_list)
    success = check(rsts, test_config["targets"])
    return success

# TODO
@test_op_wrapper
def test_deformable_transformer_encoder_layer(config):
    test_config = config["deformable_transformer_encoder_layer"]
    data_list = generate_data(test_config["data_config"])


@test_op_wrapper
def test_ms_deform_attn(config):
    test_config = config["ms_deform_attn"]
    data_list = generate_data(test_config["data_config"])
    op = eval("MSDeformAttn")
    op_config = check_config(test_config["op_config"])
    # rsts = debug_op(op, op_config, data_list)
    rsts = run_op(op, op_config, data_list)
    success = check(rsts, test_config["targets"])
    return success


if __name__ == "__main__":
    device_num = None
    rank_id = None
    ms.set_context(device_id=7, mode=1)
    # ms.set_context(device_target="CPU", mode=1)
    set_seed(42)

    # test_position_embedding_2d() # Success
    # test_ms_deform_attn_core() # Success

    # test_deformable_transformer_encoder_layer() # TODO

    test_ms_deform_attn()