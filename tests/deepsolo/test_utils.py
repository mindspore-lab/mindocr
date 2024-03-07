from functools import wraps

import numpy as np
import mindspore as ms
from mindspore import Tensor

from numpy.random import Generator, PCG64
from mindocr.models.utils.deepsolo.deepsolo_layers import *
from mindocr.models.utils.deepsolo.deformable_transformer import *

from inspect import isfunction

import yaml

### ------------ common ------------ 
def data_summary(data):
    if isinstance(data, np.ndarray):
        summary = {"sum": data.sum(), "mean": data.mean(), "std": data.std(), "dtype": data.dtype, "shape": data.shape}
        return summary
    elif isinstance(data, Tensor):
        if data.dtype == ms.bool_:
            summary = {"sum": data.sum().value(), "dtype": data.dtype, "shape": data.shape}
        else:
            summary = {"sum": data.sum().value(), "mean": data.mean().value(), "std": data.std().value(), "dtype": data.dtype, "shape": data.shape}
        return summary
    else:
        raise TypeError("Data Type Error")


# Return data: List[samples, input_num], item is Tensor
def generate_data(data_config):
    data_list = []
    for dc in data_config:
        func = eval(dc["func"])
        dc.pop("func")
        data = func(**dc)
        data_list.append(data)
    return data_list


def debug_op(op, op_config, data_list):
    if isfunction(op):
        op_instance = op
    else:
        op_instance = op(**op_config)
    rsts = []
    print(f"----- debug op:{op.__name__} ----")
    for i, data in enumerate(data_list):
        print(f"sample {i}")
        print("    input")
        for j, d in enumerate(data):
            summary = data_summary(d)
            print(f"        input_{j}: {summary}")
        rst = op_instance(*data)
        print("    output")
        if isinstance(rst, ms.Tensor):
            summary = data_summary(rst)
            rst = [rst]
            print(f"        output: {summary}")
        elif isinstance(rst, list):
            for j, r in enumerate(rst):
                summary = data_summary(r)
                print(f"        output_{j}: {summary}")
        elif isinstance(rst, dict):
            for k, v in rst.items():
                summary = data_summary(v)
                print(f"        {k}: {summary}")
        else:
            None
        rsts.append(rst)
    print(f"--------------------------------")
    return rsts

def run_op(op, op_config, data_list):
    if isfunction(op):
        op_instance = op
    else:
        op_instance = op(**op_config)
    rsts = []
    for data in data_list:
        rst = op_instance(*data)
        if isinstance(rst, ms.Tensor):
            rst = [rst]
        rsts.append(rst)
    return rsts

def check(rsts, targets):
    def check_value(target, check_out):
        if isinstance(target, (list, tuple)):
            if len(check_out) != len(target):
                return False
            for i in range(len(check_out)):
                if check_out[i] != target[i]:
                    return False
        else:
            if ops.abs((check_out - target) / target).value() > 1e-3:
                return False
        return True
    
    def single_check(tgt, out, i, j):
        if_success = True
        for check_op, v in tgt.items():
            check_func = eval(check_op)
            if check_op == "ms.Tensor.item":
                check_out = check_func(out, index=tuple(v["param"]["index"]))
                if not check_value(v["target"], check_out):
                    print(f"Check {check_op} Failed: Sample:{i}, Output:{j}, rst:{check_out}, target:{v['target']}")
                    if_success = False
                continue

            if not "param" in v.keys():
                check_out = check_func(out)
                if not check_value(v["target"], check_out):
                    print(f"Check {check_op} Failed: Sample:{i}, Output:{j}, rst:{check_out}, target:{v['target']}")
                    if_success = False
            else:
                check_out = check_func(out, **v["param"])
                if not check_value(v["target"], check_out):
                    print(f"Check {check_op} Failed: Sample:{i}, Output:{j}, rst:{check_out}, target:{v['target']}")
                    if_success = False
        return if_success
    
    if_success = True
    check_samples = zip(rsts, targets)
    for i, check_sample in enumerate(check_samples):
        # for each samples
        rst, target = check_sample
        if len(rst) != len(target):
            raise ValueError("len(rst) should equal to len(tgt)")
        for j in range(len(rst)):
            # for each output
            output = rst[j]
            tgt = target[j]
            if_success = single_check(tgt, output, i, j)
    return if_success

def check_config(op_config):
    for k, v in op_config.items():
        if v == "None":
            op_config[k] = None
    return op_config


### ------------ debug and run wrapper ------------ 
def test_op_wrapper(func):
    @wraps(func)
    def op_wrapper():
        with open("tests/deepsolo/test_config.yaml") as fp:
            config = yaml.safe_load(fp)
        success = func(config)
        if(success):
            print(f"Test {func.__name__} Success")
        else:
            print(f"Test {func.__name__} Fail")
    return op_wrapper

def data_generator_debug_wrapper(func):
    @wraps(func)
    def generator_wrapper(*args, **kwargs):
        print(f"----- debug {func.__name__} ----")
        print(f"args: {args}")
        print(f"kwargs: {kwargs}")
        data_dict = func(*args, **kwargs)
        print("data summary:")
        for key, value in data_dict.items():
            print(f"    {key}")
            for sub_key, sub_value in value.items():
                summary = data_summary(sub_value)
                print(f"        data: {sub_key}, {summary}")
        print(f"--------------------------------")
        return list(data_dict["output"].values())
    return generator_wrapper

def data_generator_run_wrapper(func):
    @wraps(func)
    def generator_wrapper(*args, **kwargs):
        data_dict = func(*args, **kwargs)
        return list(data_dict["output"].values())
    return generator_wrapper


### ------------ data generator -------------
data_generator_wrapper = data_generator_debug_wrapper
# data_generator_wrapper = data_generator_run_wrapper

@data_generator_wrapper
def bool_generator(seed, shape, dtype, ratio):   # success
    np.random.seed(seed)
    d = np.random.random_sample(shape)
    d_bool = np.copy(d)
    d_bool[d_bool > ratio] = True
    d_bool[d_bool <= ratio] = False
    output = Tensor.from_numpy(d_bool)
    return {"intermediate": {"random_sample": d, "d_bool": d_bool}, "output": {"_0": output}}

@data_generator_wrapper
def dense_weight_generator(seed, shape, dtype):
    np.random.seed(seed)
    d = np.random.random_sample(shape)
    output = Tensor.from_numpy(d).astype(dtype)
    return {"output": {"weight": output}}

@data_generator_wrapper
def dense_bias_generator(seed, n, dtype):
    np.random.seed(seed)
    d = np.random.random_sample([n])
    output = Tensor.from_numpy(d).astype(dtype)
    return {"output": {"bias": output}}

@data_generator_wrapper
def load_ms_deform_attn_core_data():  # success
    value_path = "tests/deepsolo/test_data/ms_deform_attn_core_data/value.npy"
    input_spatial_shapes_path = "tests/deepsolo/test_data/ms_deform_attn_core_data/input_spatial_shapes.npy"
    sampling_locations_path = "tests/deepsolo/test_data/ms_deform_attn_core_data/sampling_locations.npy"
    attention_weights_path = "tests/deepsolo/test_data/ms_deform_attn_core_data/attention_weights.npy"

    value = Tensor.from_numpy(np.load(value_path))
    input_spatial_shapes = Tensor.from_numpy(np.load(input_spatial_shapes_path))
    sampling_locations = Tensor.from_numpy(np.load(sampling_locations_path))
    attention_weights = Tensor.from_numpy(np.load(attention_weights_path))
    return {
        "output": {
            "value": value,
            "input_spatial_shapes": input_spatial_shapes,
            "sampling_locations": sampling_locations,
            "attention_weights": attention_weights
        }
    }

@data_generator_wrapper
def load_deformable_transformer_encoder_layer_data():  # success
    src_path = "tests/deepsolo/test_data/deformable_transformer_encoder_layer_data/src.npy"
    pos_path = "tests/deepsolo/test_data/deformable_transformer_encoder_layer_data/pos.npy"
    reference_points_path = "tests/deepsolo/test_data/deformable_transformer_encoder_layer_data/reference_points.npy"
    spatial_shapes_path = "tests/deepsolo/test_data/deformable_transformer_encoder_layer_data/spatial_shapes.npy"
    level_start_index_path = "tests/deepsolo/test_data/deformable_transformer_encoder_layer_data/level_start_index.npy"
    src = Tensor.from_numpy(np.load(src_path))
    pos = Tensor.from_numpy(np.load(pos_path))
    reference_points = Tensor.from_numpy(np.load(reference_points_path))
    spatial_shapes = Tensor.from_numpy(np.load(spatial_shapes_path))
    level_start_index = Tensor.from_numpy(np.load(level_start_index_path))
    return {
        "output": {
            "src": src,
            "pos": pos,
            "reference_points": reference_points,
            "spatial_shapes": spatial_shapes,
            "level_start_index": level_start_index,
        }
    }

@data_generator_wrapper
def load_ms_deform_attn_data():  # success
    query_path = "tests/deepsolo/test_data/ms_detorm_attn/query.npy"
    reference_points_path = "tests/deepsolo/test_data/ms_detorm_attn/reference_points.npy"
    input_flatten_path = "tests/deepsolo/test_data/ms_detorm_attn/input_flatten.npy"
    input_spatial_shapes_path = "tests/deepsolo/test_data/ms_detorm_attn/input_spatial_shapes.npy"
    input_level_start_index_path = "tests/deepsolo/test_data/ms_detorm_attn/input_level_start_index.npy"
    input_padding_mask_path = "tests/deepsolo/test_data/ms_detorm_attn/input_padding_mask.npy"
    query = Tensor.from_numpy(np.load(query_path))
    reference_points = Tensor.from_numpy(np.load(reference_points_path))
    input_flatten = Tensor.from_numpy(np.load(input_flatten_path))
    input_spatial_shapes = Tensor.from_numpy(np.load(input_spatial_shapes_path))
    input_level_start_index = Tensor.from_numpy(np.load(input_level_start_index_path))
    input_padding_mask = Tensor.from_numpy(np.load(input_padding_mask_path))
    return {
        "output": {
            "query": query,
            "reference_points": reference_points,
            "input_flatten": input_flatten,
            "input_spatial_shapes": input_spatial_shapes,
            "input_level_start_index": input_level_start_index,
            "input_padding_mask": input_padding_mask,
        }
    }