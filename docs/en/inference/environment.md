## Offline Inference Environment Installation

This tutorial only covers the environment installation of MindOCR for offline inference on Atlas 300 series inference devices.

### 1. Version Matching Table

Please refer to the version matching table when setting up the inference environment. It is recommended to use MindSpore 2.2.14 for inference.

The version of Driver and Firmware is different for different chips. Please download the matched [driver and firmware](https://www.hiascend.com/en/hardware/firmware-drivers/community?product=2&model=3&cann=7.0.0.beta1&driver=1.0.22.alpha) according to the [CANN package](https://www.hiascend.com/en/software/cann/community) version.

Now, we taking Atlas 300I Inference Card (Model: 3010) run on x86 CPU as an example, introduce version matching relationship. The following installation version is also introduced using this as an example.

| MindSpore | Driver | Firmware | CANN | MindOCR |
| --- | --- | --- | --- | --- |
| 2.2.14 | 23.0.0 | 7.1.0.3.220 | 7.0.0.beta1 | v0.4.0 |

**Other MindSpore and Ascend software version matching please refer to [MindSpore Install](https://www.mindspore.cn/install).**

### 2. Ascend Environment Installation

There are two versions of the Ascend software package, the commercial edition and the community edition. The commercial edition is only for commercial customers and download is restricted; The community edition can be freely downloaded, and the following examples all use the community edition.

This example uses the Ascend package that comes with MindSpore 2.2.14, other MindSpore version please refer to [Installing Ascend AI processor software package](https://www.mindspore.cn/install/en#installing-ascend-ai-processor-software-package).

| software | version | package name | download |
| --- | --- | --- | --- |
| Driver | 23.0.0 | A300-3010-npu-driver_23.0.0_linux-x86_64.run | [link](https://www.hiascend.com/en/hardware/firmware-drivers/community?product=2&model=3&cann=7.0.0.beta1&driver=1.0.22.alpha) |
| Firmware | 7.1.0.3.220 | A300-3010-npu-firmware_7.1.0.3.220.run | [link](https://www.hiascend.com/en/hardware/firmware-drivers/community?product=2&model=3&cann=7.0.0.beta1&driver=1.0.22.alpha) |
| CANN nnae | 7.0.0.beta1 | Ascend-cann-nnae_7.0.0_linux-x86_64.run | [link](https://www.hiascend.com/developer/download/community/result?module=cann&cann=7.0.0.beta1) |
| CANN kernels(Optional) | 7.0.0.beta1 | Ascend-cann-kernels-310p_7.0.0_linux.run | [link](https://www.hiascend.com/developer/download/community/result?module=cann&cann=7.0.0.beta1) |

#### Install

```shell
# Note: When installing a new machine, install the driver first and then the firmware.
# When the scenario of upgrade, install the firmware first and then the driver.
bash A300-3010-npu-driver_23.0.0_linux-x86_64.run --full
bash A300-3010-npu-firmware_7.1.0.3.220.run --full
bash Ascend-cann-nnae_7.0.0_linux-x86_64.run --install
bash Ascend-cann-kernels-310p_7.0.0_linux.run --install

pip uninstall te topi hccl -y
pip install sympy
pip install /usr/local/Ascend/nnae/latest/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/nnae/latest/lib64/hccl-*-py3-none-any.whl
reboot
```

#### Configure Environment Variables

```shell
source /usr/local/Ascend/nnae/set_env.sh
```

### 3. MindSpore Install

```shell
pip install mindspore==2.2.14

# Check version number, offline inference MindSpore only uses CPU
python -c "import mindspore;mindspore.set_context(device_target='CPU');mindspore.run_check()"
```

### 4. MindSpore Lite Install

| software | version | package name | download |
| --- | --- | --- | --- |
| Inference Toolkit | 2.2.14 | mindspore-lite-2.2.14-linux-{arch}.tar.gz | [link](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html#2-2-14) |
| Python Wheel | 2.2.14 | mindspore_lite-2.2.14-{python_version}-linux_{arch}.whl | [link](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html#2-2-14) |

Unzip the inference toolkit and pay attention to setting environment variables:

```shell
tar -xvf mindspore-lite-2.2.14-linux-{arch}.tar.gz
cd mindspore-lite-2.2.14-linux-{arch}/
export LITE_HOME=${PWD}    # The actual path after extracting the tar package
export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/runtime/third_party/dnnl:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
export PATH=$LITE_HOME/tools/converter/converter:$LITE_HOME/tools/benchmark:$PATH
```

If interface with python, install the required whl package using pip.

```shell
pip install mindspore_lite-2.2.14-{python_version}-linux_{arch}.whl
```

### 5. MindOCR Install

```shell
git clone https://github.com/mindspore-lab/mindocr.git
cd mindocr
pip install -e .
```

> Using '- e' to enter editable mode, help solve import issues.
