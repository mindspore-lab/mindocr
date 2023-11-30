import sys
import os

import pytest

sys.path.append("../../")
import subprocess
from abc import ABCMeta,abstractmethod
from collections import defaultdict

sys.path.append("data")

from data_for_export import data_info_for_converte_static_model_from_download_mindir
from data_for_export import data_info_for_converte_static_model_from_exported_mindir
from data_for_export import data_info_for_converte_dynamic_model_from_exported_mindir

from data_for_export import data_info_for_export_static_model
from data_for_export import data_info_for_export_dynamic_model

output_path = "./output"

path_exported_static_model = "/home/zhq/code/codehub/models/models_from_205_ms22/export_static_rst"
path_exported_dynamic_model = "/home/zhq/code/codehub/models/models_from_205_ms22/export_dynamic_rst"


class BaseTestConvertModel(metaclass=ABCMeta):
    def __init__(self, convert_tool, benchmark_tool, models_info, save_path, log_file, config_file) -> None:
        self.convert_tool = convert_tool
        self.benchmark_tool = benchmark_tool
        self.models_info = models_info
        self.save_path = save_path
        self.log_file = log_file
        self.config_file = config_file

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.log_handle = open(self.log_file, 'w')
        self.record = {}

    def generate_static_shape_config_file(self, var, data_shape):
        with open(self.config_file, 'w') as f:
            f.write("[ascend_context]\n")
            f.write("input_format=NCHW\n")
            f.write(f"input_shape={var}:[{data_shape}]")

    def generate_dynamic_shape_config_file(self, var, data_shape):
        with open(self.config_file, 'w') as f:
            f.write("[acl_build_options]\n")
            f.write("input_format=NCHW\n")
            f.write(f"input_shape_range={var}:[{data_shape}]")

    def convert_mindir(self, model, info, mindir_path):
        converted_model_path = os.path.join(self.save_path, model + f'_{info}')
        command = f"{self.convert_tool} --fmk=MINDIR --modelFile={mindir_path} --outputFile={converted_model_path} --optimize=ascend_oriented --configFile={self.config_file}"
        print(f"\033[34mConvert Command\033[0m: {command}")
        subprocess.call(f"echo Convert Command: {command}".split(), stdout=self.log_handle, stderr=self.log_handle)
        ret = subprocess.call(command.split(), stdout=self.log_handle, stderr=self.log_handle)
        if ret == 0:
            print(f"\033[32mConvert Success\033[0m: {converted_model_path}.mindir")
            subprocess.call(f"echo Convert Success: {converted_model_path}.mindir".split(), stdout=self.log_handle, stderr=self.log_handle)
        else:
            print("\033[31mConvert Failed \033[0m")y
            subprocess.call(f"echo Convert Failed: {converted_model_path}.mindir".split(), stdout=self.log_handle, stderr=self.log_handle)
        return ret

    def infer_static_shape_ascend(self, converted_model_path, data_shape):
        command = f"{self.benchmark_tool} --modelFile={converted_model_path} --device=Ascend --inputShapes={data_shape} --loopCount=100 --warmUpLoopCount=10"
        print(f"\033[34mBenchmark Command\033[0m: {command}")
        subprocess.call(f"echo Benchmark Command: {command}".split(), stdout=self.log_handle, stderr=self.log_handle)
        ret = subprocess.call(command.split(), stdout=self.log_handle, stderr=self.log_handle)
        if ret == 0:
            print("\033[32mInfer Static Shape Success \033[0m")
            subprocess.call(f"echo Infer Static Shape Success".split(), stdout=self.log_handle, stderr=self.log_handle)
        else:
            print("\033[31mInfer Static Shape Failed \033[0m")
            subprocess.call(f"echo Infer Static Shape Failed".split(), stdout=self.log_handle, stderr=self.log_handle)
        return ret

    def infer_dynamic_shape_ascend(self, converted_model_path, data_shape_list):
        rets = []
        for data_shape in data_shape_list:
            command = f"{self.benchmark_tool} --modelFile={converted_model_path} --device=Ascend --inputShapes={data_shape} --loopCount=100 --warmUpLoopCount=10"
            print(f"\033[34mBenchmark Command\033[0m: {command}")
            subprocess.call(f"echo Benchmark Command: {command}".split(), stdout=self.log_handle, stderr=self.log_handle)
            ret = subprocess.call(command.split(), stdout=self.log_handle, stderr=self.log_handle)
            if ret == 0:
                print("\033[32mInfer Dynamic Shape Success \033[0m")
                subprocess.call("echo Infer Dynamic Shape Success".split(), stdout=self.log_handle, stderr=self.log_handle)
            else:
                print("\033[31mInfer Dynamic Shape Failed \033[0m")
                subprocess.call("echo Infer Dynamic Shape Failed".split(), stdout=self.log_handle, stderr=self.log_handle)
            rets.append(ret)
        return rets

    def run(self):
        count = 0
        for model in self.models_info.keys():
            print(f"\033[36mTesting model:{model}, {count+1}/{len(self.models_info)}\033[0m")
            self.run_single(model)
            count += 1

    def __del__(self):
        self.log_handle.close()

    @abstractmethod
    def run_single(self, model):
        print('Func run_single is an abstract function.')

class TestConvertStaticModelFromDownloadMindir(BaseTestConvertModel):
    def __init__(self, convert_tool, benchmark_tool, models_info, download_path, save_path, log_file, config_file) -> None:
        super(TestConvertStaticModelFromDownloadMindir, self).__init__(convert_tool, benchmark_tool, models_info, save_path, log_file, config_file)
        self.download_path = download_path
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

    def run_single(self, model):
        info = self.models_info[model]
        self.record[model] = {'Static Convert': False, 'Static Benchmark': False, 'Static Shape': info['data_shape']}
        _, download_mindir_path = self.mindir_download(info['mindir_name'], info['mindir_url'])
        self.generate_static_shape_config_file(info['var'], info['data_shape'])
        ret = self.convert_mindir(model, self.info, download_mindir_path)
        if ret == 0:
            self.record[model]['Static Convert'] = True
            converted_model_path = os.path.join(self.save_path, model + f'_{self.info}.mindir')
            ret = self.infer_static_shape_ascend(converted_model_path, info['data_shape'])
            if ret == 0:
                self.record[model]['Static Benchmark'] = True
        print()
    
    def report(self):
        from prettytable import PrettyTable
        table = PrettyTable(['Convert Static Model From Download Mindir','Static Convert','Static Benchmark','Benchmark Infer Shape'])
        for model, info in self.record.items():
            static_convert = "\033[32mSuccess\033[0m" if info['Static Convert'] else "\033[31mFailed\033[0m"
            static_benchmark = "\033[32mSuccess\033[0m" if info['Static Benchmark'] else "\033[31mFailed\033[0m"
            table.add_row([model, static_convert, static_benchmark, info['Static Shape']])
        print(table)

class TestConvertStaticModelFromExportedMindir(BaseTestConvertModel):
    def __init__(self, convert_tool, benchmark_tool, models_info, exported_path, save_path, log_file, config_file) -> None:
        super(TestConvertStaticModelFromExportedMindir, self).__init__(convert_tool, benchmark_tool, models_info, save_path, log_file, config_file)
        self.exported_path = exported_path
        self.info = "lite_static"

    def run_single(self, model):
        info = self.models_info[model]
        self.record[model] = {'Static Convert': False, 'Static Benchmark': False, 'Static Shape': info['data_shape']}
        self.generate_static_shape_config_file(info['var'], info['data_shape'])
        exported_mindir_path = os.path.join(self.exported_path, info['mindir_name'])
        ret = self.convert_mindir(model, self.info, exported_mindir_path)
        if ret == 0:
            self.record[model]['Static Convert'] = True
            converted_model_path = os.path.join(self.save_path, model + f'_{self.info}.mindir')
            ret = self.infer_static_shape_ascend(converted_model_path, info['data_shape'])
            if ret == 0:
                self.record[model]['Static Benchmark'] = True
        print()
    
    def report(self):
        from prettytable import PrettyTable
        table = PrettyTable(['Convert Static Model From Exported Mindir','Static Convert','Static Benchmark','Benchmark Infer Shape'])
        for model, info in self.record.items():
            static_convert = "\033[32mSuccess\033[0m" if info['Static Convert'] else "\033[31mFailed\033[0m"
            static_benchmark = "\033[32mSuccess\033[0m" if info['Static Benchmark'] else "\033[31mFailed\033[0m"
            table.add_row([model, static_convert, static_benchmark, info['Static Shape']])
        print(table)

class TestConvertDynamicModelFromExportedMindir(BaseTestConvertModel):
    def __init__(self, convert_tool, benchmark_tool, models_info, exported_path, save_path, log_file, config_file) -> None:
        super(TestConvertDynamicModelFromExportedMindir, self).__init__(convert_tool, benchmark_tool, models_info, save_path, log_file, config_file)
        self.exported_path = exported_path
        self.infer_max_num = 0
        self.info = "lite_dynamic"

    def run_single(self, model):
        info = self.models_info[model]
        self.record[model] = defaultdict(lambda : ("", 0))
        self.record[model]['Dynamic Convert'] = False
        self.generate_dynamic_shape_config_file(info['var'], info['data_shape'])
        exported_mindir_path = os.path.join(self.exported_path, info['mindir_name'])
        ret = self.convert_mindir(model, self.info, exported_mindir_path)
        self.infer_max_num = max(self.infer_max_num, len(info['infer_shape_list']))
        if ret == 0:
            self.record[model]['Dynamic Convert'] = True
            converted_model_path = os.path.join(self.save_path, model + f'_{self.info}.mindir')
            rets = self.infer_dynamic_shape_ascend(converted_model_path, info['infer_shape_list'])
            for i, ret in enumerate(rets):
                self.record[model][f'Infer Shape {i}'] = (info['infer_shape_list'][i], ret)
        print()
    
    def report(self):
        from prettytable import PrettyTable
        tableList = ['Convert Dynamic Model From Exported Mindir','Dynamic Convert']
        for n in range(self.infer_max_num):
            tableList.append(f"Infer Shape {n}")
                     
        table = PrettyTable(tableList)
        for model, info in self.record.items():
            dynamic_convert = "\033[32mSuccess\033[0m" if info['Dynamic Convert'] else "\033[31mFailed\033[0m"
            content = [model, dynamic_convert]
            for n in range(self.infer_max_num):
                shape, ret = info[f'Infer Shape {n}']
                infer_shape = f"\033[32m{shape}\033[0m" if ret==0 else f"\033[31m{shape}\033[0m"
                content.append(infer_shape)
            table.add_row(content)
        print(table)

class BaseTestExportModel(metaclass=ABCMeta):
    def __init__(self, models_info, save_path, log_file) -> None:
        self.models_info = models_info
        self.save_path = save_path
        self.log_file = log_file

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.log_handle = open(self.log_file, 'w')
        self.record = {}

    def export_mindir(self, model, is_dynamic, data_shape_h_w, model_type):
        exported_model_path = os.path.join(self.save_path, 'model.mindir')
        command = f"bash ../../tools/export_tool.sh -c={model} -d={self.save_path} -D={is_dynamic} -H={data_shape_h_w[0]} " + \
            f"-W={data_shape_h_w[1]} -T={model_type}"
        subprocess.call(f"echo Export Command: {command}".split(), stdout=self.log_handle, stderr=self.log_handle)
        print(f"\033[34mExport Command\033[0m: {command}")
        ret = subprocess.call(command.split(), stdout=self.log_handle, stderr=self.log_handle)
        
        if ret == 0:
            print(f"\033[32mExport Success\033[0m: {exported_model_path}")
            self.log_handle.write(f"Export Success: {exported_model_path}")
        else:
            print("\033[31mExport Failed \033[0m")
            self.log_handle.write("Export Failed")
        return ret

    def run(self):
        count = 0
        for model in self.models_info.keys():
            print(f"\033[36mTesting model:{model}, {count+1}/{len(self.models_info)}\033[0m")
            self.run_single(model)
            count += 1

    def __del__(self):
        self.log_handle.close()

    @abstractmethod
    def run_single(self, model):
        print('Func run_single is an abstract function.')

class TestExportStaticModel(BaseTestExportModel):
    def __init__(self, models_info, save_path, log_file) -> None:
        super(TestExportStaticModel, self).__init__(models_info, save_path, log_file)

    def run_single(self, model):
        info = self.models_info[model]
        self.record[model] = defaultdict(lambda : ("", 0))
        self.record[model]['Static Export'] = False
        ret = self.export_mindir(model, False, info['data_shape_h_w'], "")
        if ret == 0:
            self.record[model]['Static Export'] = True

    def report(self):
        from prettytable import PrettyTable
        table = PrettyTable(['Export Static Model','Static Export'])
        for model, info in self.record.items():
            static_export = "\033[32mSuccess\033[0m" if info['Static Export'] else "\033[31mFailed\033[0m"
            table.add_row([model, static_export])
        print(table)
    
class TestExportDynamicModel(BaseTestExportModel):
    def __init__(self, models_info, save_path, log_file) -> None:
        super(TestExportDynamicModel, self).__init__(models_info, save_path, log_file)

    def run_single(self, model):
        info = self.models_info[model]
        self.record[model] = defaultdict(lambda : ("", 0))
        self.record[model]['Dynamic Export'] = False
        ret = self.export_mindir(model, True, [-1, -1], info['model_type'])
        if ret == 0:
            self.record[model]['Dynamic Export'] = True

    def report(self):
        from prettytable import PrettyTable
        table = PrettyTable(['Export Dynamic Model','Dynamic Export'])
        for model, info in self.record.items():
            dynamic_export = "\033[32mSuccess\033[0m" if info['Dynamic Export'] else "\033[31mFailed\033[0m"
            table.add_row([model, dynamic_export])
        print(table)

@pytest.mark.parametrize("output_path", output_path)
def test_convert_static_model_from_download_mindir(output_path):
    convert_tool = "converter_lite"
    benchmark_tool = "benchmark"
    save_path = f"{output_path}/convert_static_from_download"
    download_path = f"{save_path}/download"
    log_file = f"{save_path}/log.log"
    config_file = f"{save_path}/static_config.txt"
    testcase = TestConvertStaticModelFromDownloadMindir(convert_tool, benchmark_tool, data_info_for_converte_static_model_from_download_mindir, \
                                             download_path, save_path, log_file, config_file)
    testcase.run()
    testcase.report()

@pytest.mark.parametrize("output_path", output_path)
@pytest.mark.parametrize("path_exported_static_model", path_exported_static_model)
def test_convert_static_model_from_exported_mindir(output_path, path_exported_static_model):
    convert_tool = "converter_lite"
    benchmark_tool = "benchmark"
    save_path = f"{output_path}/convert_static_from_exported"
    exported_path = path_exported_static_model
    log_file = f"{save_path}/log.log"
    config_file = f"{save_path}/static_config.txt"
    testcase = TestConvertStaticModelFromExportedMindir(convert_tool, benchmark_tool, data_info_for_converte_static_model_from_exported_mindir, \
                                             exported_path, save_path, log_file, config_file)
    testcase.run()
    testcase.report()

@pytest.mark.parametrize("output_path", output_path)
@pytest.mark.parametrize("path_exported_dynamic_model", path_exported_dynamic_model)
def test_convert_dynamic_model_from_exported_mindir(output_path, path_exported_dynamic_model):
    convert_tool = "converter_lite"
    benchmark_tool = "benchmark"
    save_path = f"{output_path}/convert_dynamic_from_exported"
    exported_path = path_exported_dynamic_model
    log_file = f"{save_path}/log.log"
    config_file = f"{save_path}/dynamic_config.txt"
    testcase = TestConvertDynamicModelFromExportedMindir(convert_tool, benchmark_tool, data_info_for_converte_dynamic_model_from_exported_mindir, \
                                             exported_path, save_path, log_file, config_file)
    testcase.run()
    testcase.report()

@pytest.mark.parametrize("output_path", output_path)
def test_export_static_model(output_path):
    save_path = f"{output_path}/export_static"
    log_file = f"{save_path}/log.log"
    testcase = TestExportStaticModel(data_info_for_export_static_model, save_path, log_file)
    testcase.run()
    testcase.report()

@pytest.mark.parametrize("output_path", output_path)
def test_export_dynamic_model(output_path):
    save_path = f"{output_path}/export_dynamic"
    log_file = f"{save_path}/log.log"
    testcase = TestExportDynamicModel(data_info_for_export_dynamic_model, save_path, log_file)
    testcase.run()
    testcase.report()


if __name__ == "__main__":
    test_convert_static_model_from_download_mindir(output_path)

    test_convert_static_model_from_exported_mindir(output_path, path_exported_static_model)

    test_convert_dynamic_model_from_exported_mindir(output_path, path_exported_dynamic_model)

    test_export_static_model(output_path)

    test_export_dynamic_model(output_path)

    # export static model. e.g. dbnet_resnet50
    save_path = f"{output_path}/export_static"                 # path to save exported static model
    log_file = f"{save_path}/log.log"                          # file to store log
    model_name = "dbnet_resnet50"                              # which model to be exported
    model_exporter = TestExportStaticModel(data_info_for_export_static_model, save_path, log_file)
    model_exporter.run_single(model_name)
    model_exporter.report(model_name)


    # export dynamic model. e.g. dbnet_resnet50
    save_path = f"{output_path}/export_dynamic"                # path to save exported dynamic model
    log_file = f"{save_path}/log.log"                          # file to store log
    model_name = "dbnet_resnet50"                              # which model to be exported
    model_exporter = TestExportDynamicModel(data_info_for_export_dynamic_model, save_path, log_file)
    model_exporter.run_single(model_name)
    model_exporter.report()


    # convert static model from web
    convert_tool = "converter_lite"                            # path to converter_lite
    benchmark_tool = "benchmark"                               # path to benchmark
    save_path = f"{output_path}/convert_static_from_download"  # path to save converted static model
    download_path = f"{save_path}/download"                    # path to save downloaded model
    log_file = f"{save_path}/log.log"                          # file to store log
    config_file = f"{save_path}/static_config.txt"             # file to store config.txt. It will be generated automatically.
    model_name = "dbnet_resnet50"                              # which model to be exported
    model_converter = TestConvertStaticModelFromDownloadMindir(convert_tool, benchmark_tool,
        data_info_for_converte_static_model_from_download_mindir, download_path, save_path, log_file, config_file)
    model_converter.run(model_name)
    model_converter.report()


    # convert static model from exported model
    convert_tool = "converter_lite"                            # path to converter_lite
    benchmark_tool = "benchmark"                               # path to benchmark
    save_path = f"{output_path}/convert_static_from_download"  # path to save converted static model
    exported_path = "path/to/exported/path"                    # path to exported model
    log_file = f"{save_path}/log.log"                          # file to store log
    config_file = f"{save_path}/static_config.txt"             # file to store config.txt. It will be generated automatically.
    model_name = "dbnet_resnet50"                              # which model to be exported
    model_converter = TestConvertStaticModelFromExportedMindir(convert_tool, benchmark_tool,
        data_info_for_converte_static_model_from_exported_mindir, download_path, save_path, log_file, config_file)
    model_converter.run(model_name)
    model_converter.report()

    # convert dynamic model from exported model
    convert_tool = "converter_lite"                            # path to converter_lite
    benchmark_tool = "benchmark"                               # path to benchmark
    save_path = f"{output_path}/convert_dynamic_from_download"  # path to save converted dynamic model
    exported_path = "path/to/exported/path"                    # path to exported model
    log_file = f"{save_path}/log.log"                          # file to store log
    config_file = f"{save_path}/dynamic_config.txt"             # file to store config.txt. It will be generated automatically.
    model_name = "dbnet_resnet50"                              # which model to be exported
    model_converter = TestConvertDynamicModelFromExportedMindir(convert_tool, benchmark_tool,
        data_info_for_converte_dynamic_model_from_exported_mindir, download_path, save_path, log_file, config_file)
    model_converter.run(model_name)
    model_converter.report()