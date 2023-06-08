import logging
import subprocess
import time


class ATCConverter:
    def __init__(self, params):
        try:
            subprocess.Popen(["atc"], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as file_error:
            logging.error(
                "atc tools is not available. please set environment variables of ascend-toolkit and " "ensure valid."
            )
            raise file_error
        finally:
            time.sleep(1)

        self.atc_params = {
            "framework": 5,
            "input_format": "ND",
            "input_shape": f"{params.input_name}:{params.input_shape}",
            "soc_version": params.soc_version,
            "log": "error",
        }

    def convert_async(self, scaling, model_path, output_path):
        self.atc_params["model"] = model_path
        atc_cmd = ["atc"]
        for k, v in self.atc_params.items():
            atc_cmd.extend([f"--{k}", f"{v}"])
        if scaling:
            atc_cmd.extend(["--dynamic_dims", scaling])
        atc_cmd.extend(["--output", output_path])

        logging.info(" ".join(atc_cmd))
        subp = subprocess.Popen(atc_cmd, shell=False)
        time.sleep(1)
        return subp
