import sys
import os

import pytest

sys.path.append("../")
from tools.export_convert_tool import *  # noqa
from tools.data_for_export import *  # noqa

output_path = "./output"

path_exported_static_model = "path/to/exported_static/path"
path_exported_dynamic_model = "path/to/exported_dynamic/path"

@pytest.mark.parametrize("output_path", output_path)
def test_convert_static_model_from_download_mindir(output_path):
    convert_tool = "converter_lite"
    benchmark_tool = "benchmark"
    save_path = f"{output_path}/convert_static_from_download"
    download_path = f"{save_path}/download"
    log_file = f"{save_path}/log.log"
    config_file = f"{save_path}/static_config.txt"
    testcase = ConvertStaticModelFromDownloadMindir(convert_tool, benchmark_tool, data_info_for_converte_static_model_from_download_mindir, \
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
    testcase = ConvertStaticModelFromExportedMindir(convert_tool, benchmark_tool, data_info_for_converte_static_model_from_exported_mindir, \
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
    testcase = ConvertDynamicModelFromExportedMindir(convert_tool, benchmark_tool, data_info_for_converte_dynamic_model_from_exported_mindir, \
                                             exported_path, save_path, log_file, config_file)
    testcase.run()
    testcase.report()

@pytest.mark.parametrize("output_path", output_path)
def test_export_static_model(output_path):
    save_path = f"{output_path}/export_static"
    log_file = f"{save_path}/log.log"
    testcase = ExportStaticModel(data_info_for_export_static_model, save_path, log_file)
    testcase.run()
    testcase.report()

@pytest.mark.parametrize("output_path", output_path)
def test_export_dynamic_model(output_path):
    save_path = f"{output_path}/export_dynamic"
    log_file = f"{save_path}/log.log"
    testcase = ExportDynamicModel(data_info_for_export_dynamic_model, save_path, log_file)
    testcase.run()
    testcase.report()

if __name__ == "__main__":
    # not support export on Ascend 310P
    test_export_static_model(output_path)

    # not support export on Ascend 310P
    test_export_dynamic_model(output_path)

    test_convert_static_model_from_download_mindir(output_path)

    test_convert_static_model_from_exported_mindir(output_path, path_exported_static_model)

    test_convert_dynamic_model_from_exported_mindir(output_path, path_exported_dynamic_model)

