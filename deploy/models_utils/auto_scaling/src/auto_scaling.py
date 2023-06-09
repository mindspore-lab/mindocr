import json
import logging
import os

from .backend import ATCConverter, LiteConverter
from .scale_analyzer import DatasetAnalyzer


def save_scaling_data_deal(data_li, input_shape):
    """
    Put scaling data and user input data together.
    """
    results = {}
    count = 1
    for data in data_li:
        result = []
        for item in data[:-1].split(";"):
            item_li = item.split(",")
            item_li.reverse()
            _shape = input_shape
            while item_li:
                dyn_dim = item_li.pop()
                _shape = _shape.replace("-1", dyn_dim, 1)
            result.append(list(map(int, _shape.split(","))))
        results[f"dyn_input{count}"] = result
        count += 1
    return results


def lite_config_deal(data, input_shape, input_name):
    """
    MindSpore Lite config file adapts to dynamic shape.
    """
    input_bs, input_c, input_h, input_w = input_shape.split(",")
    content = "[ascend_context]\ninput_format = NCHW\n"
    shape = f"input_shape = {input_name}:[{input_shape}]"
    file_name = "lite_config.txt"
    if data:
        data_li = data[:-1].split(";")
        if input_bs == "-1" and "-1" not in [input_h, input_w]:
            dyn_dims = ",".join(["[" + _data + "]" for _data in data_li])
        else:
            input_hw = f"{input_h},{input_w}"
            dyn_hw = []
            for _data in data_li:
                _data_li = _data.split(",")
                if input_bs == "-1":
                    _data_li = _data_li[1:]
                _data_li.reverse()
                _input_hw = input_hw
                while _data_li:
                    _input_hw = _input_hw.replace("-1", _data_li.pop(), 1)
                dyn_hw.append("[" + _input_hw + "]")
            dyn_dims = ",".join(dyn_hw)
            shape = f"input_shape = {input_name}:[{input_bs},{input_c},-1,-1]"
            if input_bs == "-1":
                bs = data.split(",")[0]
                shape = f"input_shape = {input_name}:[{bs},{input_c},-1,-1]"
                file_name = f"lite_config_{bs}.txt"

        content = content + shape + f"\ndynamic_dims = {dyn_dims}"
    else:
        content += shape

    return file_name, content


def combine_scaling_data(scaling_data, input_shape):
    """
    Combine scaling data of batch size、height and width.
    """
    combined_scaling = []
    bs_li, height_li, width_li = scaling_data.values()
    input_bs, _, input_h, input_w = input_shape.split(",")
    if input_bs == "-1":
        if "-1" not in [input_h, input_w]:
            combined_scaling.append(";".join(list(map(str, bs_li))) + ";")
        elif input_h == "-1" and input_w == "-1":
            for bs in bs_li:
                scaling_st = ""
                for h in height_li:
                    for w in width_li:
                        scaling_st += f"{bs},{h},{w};"
                combined_scaling.append(scaling_st)
        else:
            for bs in bs_li:
                scaling_st = ""
                for h_or_w in height_li if input_h == "-1" else width_li:
                    scaling_st += f"{bs},{h_or_w};"
                combined_scaling.append(scaling_st)
    else:
        scaling_st = ""
        if input_h == "-1" and input_w == "-1":
            for h in height_li:
                for w in width_li:
                    scaling_st += f"{h},{w};"
        else:
            for h_or_w in height_li if input_h == "-1" else width_li:
                scaling_st += f"{h_or_w};"
        combined_scaling.append(scaling_st)

    return combined_scaling


def auto_scaling_process(_args, config, sys_path):
    """
    Auto-scaling pipeline.
    """
    subps = []
    input_bs, _, input_h, input_w = _args.input_shape.split(",")
    model_path = _args.model_path
    if not os.path.isfile(model_path):
        raise FileNotFoundError("model_path must be a file")
    if "-1" not in [input_bs, input_h, input_w]:
        logging.info(f"Static input shape: {_args.input_shape}, will skip auto-scaling process")
        scaling_li = [""]
    else:
        if not _args.dataset_path:
            logging.info("Auto scaling will use default values.")
            scaling_li = combine_scaling_data(config.auto_scaling.default_scaling, _args.input_shape)
        else:
            data_analyzer = DatasetAnalyzer(_args, config)
            scaling_li = combine_scaling_data(data_analyzer.start_analyzer(), _args.input_shape)
            # save data
            save_data = save_scaling_data_deal(scaling_li, _args.input_shape)
            scaling_li_save_path = os.path.abspath(os.path.join(sys_path, "scaling_data.json"))
            logging.info(f"Saving scaling data to {scaling_li_save_path}")
            with open(scaling_li_save_path, "w", encoding="utf-8") as json_fp:
                json.dump(save_data, fp=json_fp)
    model_name = os.path.splitext(os.path.split(model_path)[1])[0]
    output_base = f"{_args.output_path}/{model_name}"
    # converter
    for scaling in scaling_li:
        if scaling:
            output_path = output_base + f"_dynamic_bs{scaling.split(',')[0]}_hw"
            if input_bs != "-1":
                output_path = output_base + "_dynamic_hw"
            if input_bs == "-1" and "-1" not in [input_h, input_w]:
                output_path = output_base + "_dynamic_bs"
        else:
            output_path = output_base + "_static"
        if _args.backend == "atc":
            # ATC
            converter_atc = ATCConverter(_args)
            subps.append(converter_atc.convert_async(scaling, model_path, output_path))
        else:
            # Lite
            os.makedirs(_args.output_path, exist_ok=True)
            file_name, content = lite_config_deal(scaling, _args.input_shape, _args.input_name)
            config_file = os.path.abspath(os.path.join(sys_path, f"configs/{file_name}"))
            with open(config_file, "w", encoding="utf-8") as fp_lite:
                fp_lite.writelines(content)
            converter_lite = LiteConverter(_args)
            subps.append(converter_lite.convert_async(config_file, model_path, output_path))

    return subps
