import logging
import subprocess
import time


class ATCConverter:
    def __init__(self, params):
        try:
            subprocess.Popen(["atc"], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as e:
            logging.error("atc tools is not available. please set environment variables of ascend-toolkit and "
                          "ensure valid.")
            raise e
        finally:
            time.sleep(1)

        self.params = {"framework": 5,
                       "input_format": "ND",
                       "soc_version": params.soc_version,
                       "log": "error"
                       }

    def convert_async(self, gear, model_path, output_path):
        self.params["model"] = model_path
        atc_cmd = ["atc"]
        for k, v in self.params.items():
            atc_cmd.extend([f"--{k}", f"{v}"])
        atc_cmd.extend([f"--dynamic_dims", gear])
        atc_cmd.extend(["--output", output_path])

        print(" ".join(atc_cmd))
        subp = subprocess.Popen(atc_cmd, shell=False)
        time.sleep(1)
        return subp


class DetATCConverter(ATCConverter):
    def __init__(self, params):
        super().__init__(params)
        det_params = {"input_shape": "x:1,3,-1,-1"}
        self.params.update(det_params)


class RecATCConverter(ATCConverter):
    def __init__(self, params):
        super().__init__(params)
        rec_params = {"input_shape": f"x:-1,{params.rec_model_channel},{params.rec_model_height},-1"}
        self.params.update(rec_params)
