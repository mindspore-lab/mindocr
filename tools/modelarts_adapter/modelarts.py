import functools
import os
import shutil
import subprocess
import sys
import time
import zipfile
from typing import Any, Callable, Dict, List, Union

LOCAL_RANK = int(os.getenv("RANK_ID", 0))
INTSTALL_SUCESS_SINGAL = "/tmp/INSTALL_SUCCESS"
DATA_SUCESS_SINGAL = "/tmp/DOWNLOAD_DATA_SUCCESS"


def model_art_preprocess():
    # copy some library
    src_path = "/usr/lib64/libgeos_c.so"
    lib_dir = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "lib")
    dest_path = os.path.join(lib_dir, os.path.basename(src_path))
    if os.path.isdir(os.path.dirname(dest_path)):
        if os.path.isfile(src_path) and not os.path.isfile(dest_path):
            shutil.copy(src_path, dest_path)

    for fpath in [INTSTALL_SUCESS_SINGAL, DATA_SUCESS_SINGAL]:
        if os.path.isfile(fpath):
            os.remove(fpath)


def run_with_single_rank(local_rank: int = 0, signal: str = "/tmp/SUCCESS") -> Callable[..., Any]:
    """Run the task on 0th rank, perform synchronzation before return"""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if local_rank in [0, None]:
                result = func(*args, **kwargs)
                with open(signal, "w") as f:
                    f.write("\n")
                return result
            else:
                while not os.path.isfile(signal):
                    time.sleep(1)

        return wrapper

    return decorator


@run_with_single_rank(local_rank=LOCAL_RANK, signal=DATA_SUCESS_SINGAL)
def sync_data(s3_paths: List[str], dest: str) -> None:
    import moxing as mox

    if not os.path.isdir(dest):
        os.makedirs(dest)
    dest = os.path.abspath(dest)
    for s3_path in s3_paths:
        dest_path = os.path.join(dest, os.path.basename(s3_path))
        print(f"Dowloading data from `{s3_path}` to `{dest_path}`.")
        mox.file.copy_parallel(src_url=s3_path, dst_url=dest_path)

        if os.path.isfile(dest_path) and dest_path.endswith(".zip"):
            with zipfile.ZipFile(dest_path, "r") as zip_ref:
                zip_ref.extractall(os.path.dirname(dest_path))
            os.remove(dest_path)


@run_with_single_rank(local_rank=LOCAL_RANK, signal=INTSTALL_SUCESS_SINGAL)
def install_packages(req_path: str = "requirements.txt") -> None:
    print("INFO: Packages to be installed: ", req_path)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--retries", "20"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path, "--retries", "20"])


def modelarts_setup(args):
    if args.enable_modelarts:
        model_art_preprocess()
        cur_dir = os.path.dirname(os.path.abspath(__file__))

        # change relative path of configure file to absolute
        if not os.path.isabs(args.config):
            args.config = os.path.abspath(os.path.join(cur_dir, "../../", args.config))

        req_path = os.path.abspath(os.path.join(cur_dir, "../../requirements/modelarts.txt"))
        install_packages(req_path)
        return True
    return False


def update_config_value_by_key(config: Union[Dict, List], key: str, value: Any):
    if isinstance(config, dict):
        if key in config:
            config[key] = value
        for subconfig in config.values():
            update_config_value_by_key(subconfig, key, value)
    elif isinstance(config, list):
        for subconfig in config:
            update_config_value_by_key(subconfig, key, value)
