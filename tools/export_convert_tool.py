"""
Export/Convert tool has the following function:
    - Export ckpt to mindir file
    - Convert mindir file to lite minfir file

Args:
    model_name (str): Name of the model to be exported or converted.
    task (str): Select task to run.
    is_dynamic (bool): Whether the export or convert data shape is dynamic or static.
    input_file (str): If task is `export` or `export_convert`, input_file can be ckpt file or empty string.
        If task is `convert`, input_file can be mindir file or empty string.
        If `input_file` is empty string, default ckpt or mindir will be downloaded automatically.
    output_folder (str): The folder to save log and exported/converted file.
    convert_tool (str): The path to `benchmark` tool. It is required when `task` is `convert` or `export_convert.
         If `benchmark` can be found in environment path, this argument is not required.
    force (bool): Whether to overwrite the file(like exported/converted file) if they exist.

Example:
    >>> python export_convert_tool.py --model_name crnn_vgg7 --task export_convert --is_dynamic False \
        --output_folder output_folder --force True
    >>> python export_convert_tool.py --model_name crnn_vgg7 --task export --is_dynamic True \
        --output_folder output_folder --force True
    >>> python export_convert_tool.py --model_name svtr --task export_convert --is_dynamic False \
        --output_folder output_folder --force True
"""
import argparse
import os
import subprocess
import sys
from abc import ABCMeta, abstractmethod
from collections import defaultdict

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from mindocr import list_models  # noqa
from tools.data_for_export_convert import data_converte_dynamic_model_from_exported_mindir  # noqa
from tools.data_for_export_convert import data_converte_static_model_from_download_mindir  # noqa
from tools.data_for_export_convert import data_converte_static_model_from_exported_mindir  # noqa
from tools.data_for_export_convert import data_export_dynamic_model  # noqa
from tools.data_for_export_convert import data_export_static_model  # noqa


class BaseConvertModel(metaclass=ABCMeta):
    def __init__(self, convert_tool, benchmark_tool, models_info, save_path, log_file) -> None:
        self.convert_tool = convert_tool
        self.benchmark_tool = benchmark_tool
        self.models_info = models_info
        self.save_path = save_path
        self.log_file = os.path.join(self.save_path, log_file)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.log_handle = open(self.log_file, "w")
        self.record = {}

    def generate_static_shape_config_file(self, config_file, data_shape):
        with open(config_file, "w") as f:
            f.write("[ascend_context]\n")
            f.write("input_format=NCHW\n")
            f.write(f"input_shape={data_shape}")

    def generate_dynamic_shape_config_file(self, config_file, data_shape):
        with open(config_file, "w") as f:
            f.write("[acl_build_options]\n")
            f.write("input_format=NCHW\n")
            f.write(f"input_shape_range={data_shape}")

    def convert_mindir(self, model, info, input_file, config_file, force=False):
        converted_model_path = os.path.join(self.save_path, model + f"_{info}")
        if os.path.exists(f"{converted_model_path}.mindir"):
            if not force:
                log = f"{converted_model_path}.mindir exists. You can set `force=True` to overwrite this file."
                subprocess.call(f"echo {log}".split(), stdout=self.log_handle, stderr=self.log_handle)
                print(log)
                return False
            else:
                log = f"{converted_model_path}.mindir exists and it will be overwritten if exported successfully."
                subprocess.call(f"echo {log}".split(), stdout=self.log_handle, stderr=self.log_handle)
                os.remove(f"{converted_model_path}.mindir")
                print(log)
        command = (
            f"{self.convert_tool} --fmk=MINDIR --modelFile={input_file} --outputFile={converted_model_path}"
            + f" --optimize=ascend_oriented --configFile={config_file}"
        )
        print(f"\033[34mConvert Command\033[0m: {command}")
        subprocess.call(f"echo Benchmark Command: {command}".split(), stdout=self.log_handle, stderr=self.log_handle)
        ret = subprocess.call(command.split(), stdout=self.log_handle, stderr=self.log_handle)
        if ret == 0:
            print(f"\033[32mConvert Success\033[0m: {converted_model_path}.mindir")
            subprocess.call(
                f"echo Convert Success: {converted_model_path}.mindir".split(),
                stdout=self.log_handle,
                stderr=self.log_handle,
            )
        else:
            print("\033[31mConvert Failed \033[0m")
            subprocess.call(
                f"echo Convert Failed: {converted_model_path}.mindir".split(),
                stdout=self.log_handle,
                stderr=self.log_handle,
            )
        return ret

    def infer_static_shape_ascend(self, converted_model_path, data_shape):
        command = (
            f"{self.benchmark_tool} --modelFile={converted_model_path} --device=Ascend"
            + f" --inputShapes={data_shape} --loopCount=100 --warmUpLoopCount=10"
        )
        print(f"\033[34mBenchmark Command\033[0m: {command}")
        subprocess.call(f"echo Benchmark Command: {command}".split(), stdout=self.log_handle, stderr=self.log_handle)
        ret = subprocess.call(
            command.split(),
            stdout=self.log_handle,
            stderr=self.log_handle,
        )
        if ret == 0:
            print("\033[32mInfer Static Shape Success \033[0m")
            subprocess.call(
                "echo Infer Static Shape Success".split(),
                stdout=self.log_handle,
                stderr=self.log_handle,
            )
        else:
            print("\033[31mInfer Static Shape Failed \033[0m")
            subprocess.call(
                "echo Infer Static Shape Failed".split(),
                stdout=self.log_handle,
                stderr=self.log_handle,
            )
        return ret

    def infer_dynamic_shape_ascend(self, converted_model_path, data_shape_list):
        rets = []
        for data_shape in data_shape_list:
            command = (
                f"{self.benchmark_tool} --modelFile={converted_model_path} --device=Ascend"
                + f" --inputShapes={data_shape} --loopCount=100 --warmUpLoopCount=10"
            )
            print(f"\033[34mBenchmark Command\033[0m: {command}")
            subprocess.call(
                f"echo Benchmark Command: {command}".split(), stdout=self.log_handle, stderr=self.log_handle
            )
            ret = subprocess.call(
                command.split(),
                stdout=self.log_handle,
                stderr=self.log_handle,
            )
            if ret == 0:
                print("\033[32mInfer Dynamic Shape Success \033[0m")
                subprocess.call(
                    "echo Infer Dynamic Shape Success".split(),
                    stdout=self.log_handle,
                    stderr=self.log_handle,
                )
            else:
                print("\033[31mInfer Dynamic Shape Failed \033[0m")
                subprocess.call(
                    "echo Infer Dynamic Shape Failed".split(),
                    stdout=self.log_handle,
                    stderr=self.log_handle,
                )
            rets.append(ret)
        return rets

    def run(self, exported_path, save_config_file="temp_config.txt", models=None, force=False):
        count = 0
        if not models:
            for model in self.models_info.keys():
                info = self.models_info[model]
                print(f"\033[36mConverting model:{model}, {count+1}/{len(self.models_info)}\033[0m")
                exported_mindir_path = os.path.join(exported_path, info["mindir_name"])
                self.run_single(model, exported_mindir_path, save_config_file, force)
                count += 1
        elif isinstance(models, (tuple, list)):
            for model in models:
                info = self.models_info[model]
                print(f"\033[36mConverting model:{model}, {count+1}/{len(models)}\033[0m")
                exported_mindir_path = os.path.join(exported_path, info["mindir_name"])
                self.run_single(model, exported_mindir_path, save_config_file, force)
                count += 1
        elif isinstance(models, str):
            self.run([models], save_config_file, force)
        else:
            raise ValueError("models should be None, tuple&list containing str, or str.")

    def __del__(self):
        self.log_handle.close()
        print(f"Please refer to {self.log_file} for more details.")

    @abstractmethod
    def run_single(self, model, force=False):
        print("Func run_single is an abstract function.")


class ConvertStaticModelFromDownloadMindir(BaseConvertModel):
    def __init__(self, convert_tool, benchmark_tool, models_info, download_path, save_path, log_file) -> None:
        super(ConvertStaticModelFromDownloadMindir, self).__init__(
            convert_tool, benchmark_tool, models_info, save_path, log_file
        )
        self.download_path = os.path.join(self.save_path, download_path)
        self.info = "lite_static"
        if not os.path.exists(download_path):
            os.makedirs(download_path)

    def mindir_download(self, mindir_name, mindir_url):
        if not os.path.exists(os.path.join(self.download_path, mindir_name)):
            command = f"wget -P {self.download_path} {mindir_url} --no-check-certificate"
            subprocess.call(f"echo wget Command: {command}".split(), stdout=self.log_handle, stderr=self.log_handle)
            ret = subprocess.call(command.split(), stdout=sys.stdout, stderr=sys.stderr)
            return ret, os.path.join(self.download_path, mindir_name)
        else:
            return 0, os.path.join(self.download_path, mindir_name)

    def run_single(self, model, save_config_file, force=False):
        info = self.models_info[model]
        self.record[model] = {"Static Convert": False, "Static Benchmark": False, "Static Shape": info["data_shape"]}
        _, download_mindir_path = self.mindir_download(info["mindir_name"], info["mindir_url"])
        save_config_file = os.path.join(self.save_path, save_config_file)
        self.generate_static_shape_config_file(save_config_file, info["data_shape"])
        ret = self.convert_mindir(model, self.info, download_mindir_path, save_config_file, force)
        if ret == 0:
            self.record[model]["Static Convert"] = True
            converted_model_path = os.path.join(self.save_path, model + f"_{self.info}.mindir")
            ret = self.infer_static_shape_ascend(converted_model_path, info["infer_shape_list"][0])
            if ret == 0:
                self.record[model]["Static Benchmark"] = True
        print()

    def report(self):
        from prettytable import PrettyTable

        table = PrettyTable(
            ["Convert Static Model From Download Mindir", "Static Convert", "Static Benchmark", "Benchmark Infer Shape"]
        )
        for model, info in self.record.items():
            static_convert = "\033[32mSuccess\033[0m" if info["Static Convert"] else "\033[31mFailed\033[0m"
            static_benchmark = "\033[32mSuccess\033[0m" if info["Static Benchmark"] else "\033[31mFailed\033[0m"
            table.add_row([model, static_convert, static_benchmark, info["Static Shape"]])
        print(table)


class ConvertStaticModelFromExportedMindir(BaseConvertModel):
    def __init__(self, convert_tool, benchmark_tool, models_info, save_path, log_file) -> None:
        super(ConvertStaticModelFromExportedMindir, self).__init__(
            convert_tool, benchmark_tool, models_info, save_path, log_file
        )
        # self.exported_path = os.path.join(self.save_path, exported_path)
        self.info = "lite_static"

    def run_single(self, model, input_file, save_config_file, force=False):
        info = self.models_info[model]
        self.record[model] = {"Static Convert": False, "Static Benchmark": False, "Static Shape": info["data_shape"]}
        save_config_file = os.path.join(self.save_path, save_config_file)
        self.generate_static_shape_config_file(save_config_file, info["data_shape"])
        ret = self.convert_mindir(model, self.info, input_file, save_config_file, force)
        if ret == 0:
            self.record[model]["Static Convert"] = True
            converted_model_path = os.path.join(self.save_path, model + f"_{self.info}.mindir")
            ret = self.infer_static_shape_ascend(converted_model_path, info["infer_shape_list"][0])
            if ret == 0:
                self.record[model]["Static Benchmark"] = True
        print()

    def report(self):
        from prettytable import PrettyTable

        table = PrettyTable(
            ["Convert Static Model From Exported Mindir", "Static Convert", "Static Benchmark", "Benchmark Infer Shape"]
        )
        for model, info in self.record.items():
            static_convert = "\033[32mSuccess\033[0m" if info["Static Convert"] else "\033[31mFailed\033[0m"
            static_benchmark = "\033[32mSuccess\033[0m" if info["Static Benchmark"] else "\033[31mFailed\033[0m"
            table.add_row([model, static_convert, static_benchmark, info["Static Shape"]])
        print(table)


class ConvertDynamicModelFromExportedMindir(BaseConvertModel):
    def __init__(self, convert_tool, benchmark_tool, models_info, save_path, log_file) -> None:
        super(ConvertDynamicModelFromExportedMindir, self).__init__(
            convert_tool, benchmark_tool, models_info, save_path, log_file
        )
        self.infer_max_num = 0
        self.info = "lite_dynamic"

    def run_single(self, model, input_file, save_config_file, force=False):
        info = self.models_info[model]
        self.record[model] = defaultdict(lambda: ("", 0))
        self.record[model]["Dynamic Convert"] = False
        save_config_file = os.path.join(self.save_path, save_config_file)
        self.generate_dynamic_shape_config_file(save_config_file, info["data_shape"])
        ret = self.convert_mindir(model, self.info, input_file, save_config_file, force)
        self.infer_max_num = max(self.infer_max_num, len(info["infer_shape_list"]))
        if ret == 0:
            self.record[model]["Dynamic Convert"] = True
            converted_model_path = os.path.join(self.save_path, model + f"_{self.info}.mindir")
            rets = self.infer_dynamic_shape_ascend(converted_model_path, info["infer_shape_list"])
            for i, ret in enumerate(rets):
                self.record[model][f"Infer Shape {i}"] = (info["infer_shape_list"][i], ret)
        print()

    def report(self):
        from prettytable import PrettyTable

        tableList = ["Convert Dynamic Model From Exported Mindir", "Dynamic Convert"]
        for n in range(self.infer_max_num):
            tableList.append(f"Infer Shape {n}")

        table = PrettyTable(tableList)
        for model, info in self.record.items():
            dynamic_convert = "\033[32mSuccess\033[0m" if info["Dynamic Convert"] else "\033[31mFailed\033[0m"
            content = [model, dynamic_convert]
            for n in range(self.infer_max_num):
                shape, ret = info[f"Infer Shape {n}"]
                infer_shape = f"\033[32m{shape}\033[0m" if ret == 0 else f"\033[31m{shape}\033[0m"
                content.append(infer_shape)
            table.add_row(content)
        print(table)


class BaseExportModel(metaclass=ABCMeta):
    def __init__(self, models_info, save_path, log_file) -> None:
        self.models_info = models_info
        self.save_path = save_path
        self.log_file = os.path.join(self.save_path, log_file)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.log_handle = open(self.log_file, "w")
        self.record = {}

    def export_mindir(self, model, is_dynamic, data_shape_h_w, model_type, input_file="", force=False):
        exported_model_path = os.path.join(self.save_path, f"{model}.mindir")

        model_name = os.path.splitext(os.path.basename(model))[0]
        export_mindir_filename = f"{model_name}.mindir"
        export_mindir_path = os.path.join(self.save_path, export_mindir_filename)
        if os.path.exists(export_mindir_path):
            if not force:
                log = f"{export_mindir_path} exists. You can set `force=True` to overwrite this file."
                subprocess.call(f"echo {log}".split(), stdout=self.log_handle, stderr=self.log_handle)
                print(log)
                return False
            else:
                log = f"{export_mindir_path} exists and it will be overwritten if exported successfully."
                subprocess.call(f"echo {log}".split(), stdout=self.log_handle, stderr=self.log_handle)
                os.remove(export_mindir_path)
                print(log)
        command = f"python export.py --model_name_or_config {model} --save_dir {self.save_path}"

        if len(input_file) > 0 and os.path.exists(input_file):
            command = f"{command} --local_ckpt_path {input_file}"

        if is_dynamic:
            command = f"{command} --is_dynamic_shape {is_dynamic} --model_type {model_type}"
        else:
            command = f"{command} --is_dynamic_shape {is_dynamic} --data_shape {data_shape_h_w[0]} {data_shape_h_w[1]}"

        subprocess.call(f"echo Export Command: {command}".split(), stdout=self.log_handle, stderr=self.log_handle)
        print(f"\033[34mExport Command\033[0m: {command}")
        subprocess.call(command.split(), stdout=self.log_handle, stderr=self.log_handle)

        if os.path.exists(export_mindir_path):
            print(f"\033[32mExport Success\033[0m: {exported_model_path}")
            self.log_handle.write(f"Export Success: {exported_model_path}")
            return True
        else:
            print("\033[31mExport Failed \033[0m")
            self.log_handle.write("Export Failed")
            return False

    def run(self, models=None, force=False):
        count = 0
        if not models:
            for model in self.models_info.keys():
                print(f"\033[36mExporting model:{model}, {count+1}/{len(self.models_info)}\033[0m")
                self.run_single(model, "", force)
                count += 1
        elif isinstance(models, (tuple, list)):
            for model in models:
                print(f"\033[36mExporting model:{model}, {count+1}/{len(models)}\033[0m")
                self.run_single(model, "", force)
                count += 1
        elif isinstance(models, str):
            self.run([models], force)
        else:
            raise ValueError("models should be None, tuple&list containing str, or str.")

    def __del__(self):
        self.log_handle.close()
        print(f"Please refer to {self.log_file} for more details.")

    @abstractmethod
    def run_single(self, model, input_file="", force=False):
        print("Func run_single is an abstract function.")


class ExportStaticModel(BaseExportModel):
    def __init__(self, models_info, save_path, log_file) -> None:
        super(ExportStaticModel, self).__init__(models_info, save_path, log_file)

    def run_single(self, model, input_file="", force=False):
        info = self.models_info[model]
        self.record[model] = defaultdict(lambda: ("", 0))
        self.record[model]["Static Export"] = False
        ret = self.export_mindir(model, False, info["data_shape_h_w"], "", input_file, force)
        if ret:
            self.record[model]["Static Export"] = True

    def report(self):
        from prettytable import PrettyTable

        table = PrettyTable(["Export Static Model", "Static Export"])
        for model, info in self.record.items():
            static_export = "\033[32mSuccess\033[0m" if info["Static Export"] else "\033[31mFailed\033[0m"
            table.add_row([model, static_export])
        print(table)


class ExportDynamicModel(BaseExportModel):
    def __init__(self, models_info, save_path, log_file) -> None:
        super(ExportDynamicModel, self).__init__(models_info, save_path, log_file)

    def run_single(self, model, input_file="", force=False):
        info = self.models_info[model]
        self.record[model] = defaultdict(lambda: ("", 0))
        self.record[model]["Dynamic Export"] = False
        ret = self.export_mindir(model, True, [-1, -1], info["model_type"], input_file, force)
        if ret:
            self.record[model]["Dynamic Export"] = True

    def report(self):
        from prettytable import PrettyTable

        table = PrettyTable(["Export Dynamic Model", "Dynamic Export"])
        for model, info in self.record.items():
            dynamic_export = "\033[32mSuccess\033[0m" if info["Dynamic Export"] else "\033[31mFailed\033[0m"
            table.add_row([model, dynamic_export])
        print(table)


def run_export(model_name, is_dynamic, input_file="", export_folder="export_folder", force=False):
    if not is_dynamic:
        log_file = f"export_static_{model_name}.log"
        model_exporter = ExportStaticModel(data_export_static_model, export_folder, log_file)
    else:
        log_file = f"export_dynamic_{model_name}.log"
        model_exporter = ExportDynamicModel(data_export_dynamic_model, export_folder, log_file)
    if model_name == "all":
        model_exporter.run(force=force)
    else:
        model_exporter.run_single(model_name, input_file=input_file, force=force)
    model_exporter.report()


def run_convert(
    model_name,
    is_dynamic,
    input_file="",
    convert_folder="convert_folder",
    convert_tool="converter_lite",
    benchmark_tool="benchmark",
    force=False,
):
    if not is_dynamic:
        log_file = f"convert_static_{model_name}.log"
        config_file = f"convert_static_{model_name}.txt"
        if len(input_file) == 0:  # download mindir
            download_path = "download_path"
            model_converter = ConvertStaticModelFromDownloadMindir(
                convert_tool,
                benchmark_tool,
                data_converte_static_model_from_download_mindir,
                download_path,
                convert_folder,
                log_file,
            )
            if model_name == "all":
                model_converter.run(exported_path=input_file, force=force)
            else:
                model_converter.run_single(model_name, config_file, force=force)
            model_converter.report()
        else:
            model_converter = ConvertStaticModelFromExportedMindir(
                convert_tool,
                benchmark_tool,
                data_converte_static_model_from_exported_mindir,
                convert_folder,
                log_file,
            )
            if model_name == "all":
                model_converter.run(exported_path=input_file, force=force)
            else:
                model_converter.run_single(model_name, input_file, config_file, force=force)
            model_converter.report()
    else:
        log_file = f"convert_dynamic_{model_name}.log"
        config_file = f"convert_dynamic_{model_name}.txt"
        if len(input_file) == 0:  # download mindir
            print("Convert dynamic model from download mindir directly is not supported.")
        else:
            model_converter = ConvertDynamicModelFromExportedMindir(
                convert_tool,
                benchmark_tool,
                data_converte_dynamic_model_from_exported_mindir,
                convert_folder,
                log_file,
            )
            if model_name == "all":
                model_converter.run(exported_path=input_file, force=force)
            else:
                model_converter.run_single(model_name, input_file, config_file, force=force)
            model_converter.report()


def run_export_convert(
    model_name,
    is_dynamic,
    input_file="",
    run_folder="export_convert_folder",
    convert_tool="converter_lite",
    benchmark_tool="benchmark",
    force=False,
):
    if not os.path.exists(run_folder):
        os.path.makedirs(run_folder)
    export_folder = os.path.join(run_folder, "export_folder")
    run_export(model_name, is_dynamic, input_file, export_folder, force=force)
    if model_name == "all":
        convert_input_file = export_folder
    else:
        convert_input_file = os.path.join(export_folder, f"{model_name}.mindir")
    convert_folder = os.path.join(run_folder, "convert_folder")
    run_convert(
        model_name,
        is_dynamic,
        convert_input_file,
        convert_folder,
        convert_tool=convert_tool,
        benchmark_tool=benchmark_tool,
        force=force,
    )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "1"):
        return True
    elif v.lower() in ("no", "false", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert model checkpoint to mindir format.")
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model to be exported or converted. If `all`, all models supported will be run",
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["export", "convert", "export_convert"],
        help="The task will run",
    )

    parser.add_argument(
        "--is_dynamic",
        type=str2bool,
        default=False,
        help="Whether the export or convert data shape is dynamic or static.",
    )

    parser.add_argument(
        "--input_file",
        type=str,
        default="",
        help="If task is `export` or `export_convert`, input_file can be ckpt file or empty string. \
            If task is `convert`, input_file can be mindir file or empty string. \
            If `input_file` is empty string, default ckpt or mindir will be downloaded automatically.",
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="output_folder",
        help="The folder to save log and exported/converted file.",
    )

    parser.add_argument(
        "--convert_tool",
        type=str,
        default="converter_lite",
        help="The path to `converter_lite` tool. It is required when `task` is `convert` or `export_convert`. \
            If `converter_lite` can be found in environment path, this argument is not required.",
    )

    parser.add_argument(
        "--benchmark_tool",
        type=str,
        default="benchmark",
        help="The path to `benchmark` tool. It is required when `task` is `convert` or `export_convert`. \
            If `benchmark` can be found in environment path, this argument is not required.",
    )

    parser.add_argument(
        "--force",
        type=str2bool,
        default=False,
        help="Whether to overwrite the file(like exported/converted file) if they exist.",
    )

    parser.add_argument(
        "--show_model_list",
        action="store_true",
        default=False,
        help="List all supported models.",
    )

    args = parser.parse_args()

    if args.show_model_list:
        print("Support models:")
        print(list_models())
        exit()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    if args.task == "export":
        export_folder = os.path.join(args.output_folder, "export_folder")
        run_export(args.model_name, args.is_dynamic, args.input_file, export_folder, args.force)
    elif args.task == "convert":
        convert_folder = os.path.join(args.output_folder, "convert_folder")
        run_convert(
            args.model_name,
            args.is_dynamic,
            args.input_file,
            convert_folder,
            args.convert_tool,
            args.benchmark_tool,
            args.force,
        )
    else:
        run_export_convert(
            args.model_name,
            args.is_dynamic,
            args.input_file,
            args.output_folder,
            args.convert_tool,
            args.benchmark_tool,
            args.force,
        )
