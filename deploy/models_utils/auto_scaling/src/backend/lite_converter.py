import logging
import subprocess
import time


class LiteConverter:
    def __init__(self, params):
        try:
            subprocess.Popen(
                ["converter_lite"],
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as file_error:
            logging.error("Mindspore Lite tools is not available. please set environment variables and ensure valid.")
            raise file_error
        finally:
            time.sleep(1)

        self.lite_params = {
            "fmk": "ONNX",
            "saveType": "MINDIR",
            "NoFusion": "false",
            "device": params.soc_version[:6],
        }

    def convert_async(self, config_file, model_path, output_path):
        self.lite_params["modelFile"] = model_path
        lite_cmd = ["converter_lite"]
        for k, v in self.lite_params.items():
            lite_cmd.extend([f"--{k}={v}"])
        lite_cmd.extend([f"--configFile={config_file}"])
        lite_cmd.extend([f"--outputFile={output_path}"])

        logging.info(" ".join(lite_cmd))
        subp = subprocess.Popen(lite_cmd, shell=False)
        time.sleep(1)
        return subp
