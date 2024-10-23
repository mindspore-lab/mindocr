## 离线推理环境准备

本教程仅涉及MindOCR在Ascend310/Ascend310P设备离线推理环境准备。

### 1. 版本配套关系表

搭建推理环境请参考版本配套关系，推荐使用MindSpore 2.2.14版本进行推理。

| MindSpore | Driver | Firmware | CANN | MindOCR |
| --- | --- | --- | --- | --- |
| 2.2.14 | 配套 CANN 7.0.0.beta1 | 配套 CANN 7.0.0.beta1 | 7.0.0.beta1 | v0.4.0 |

Driver和Firmware不同芯片对应的型号不一样，请根据CANN包版本下载对应的Driver和Firmware。

**其他MindSpore版本配套关系参考[MindSpore 安装](https://www.mindspore.cn/install) ，其他MindSpore版本没有经过完整测试，谨慎使用。**

### 2. Ascend环境安装

昇腾AI处理器配套软件包有两个版本，商用版和社区版。商用版仅供商业客户使用，下载受限；社区版本可自由下载，以下例子均使用社区版本。

本章使用MindSpore2.2.14配套的Ascend配套软件包，其他版本请参考[安装昇腾AI处理器配套软件包](https://www.mindspore.cn/install#安装昇腾ai处理器配套软件包) 小节进行安装。

| software | version | package name | download |
| --- | --- | --- | --- |
| Driver | 配套 CANN 7.0.0.beta1 | {device}-npu-driver_{version}_linux-{arch}.run | [link](https://www.hiascend.com/hardware/firmware-drivers/community?product=2&model=2&cann=7.0.0.beta1&driver=1.0.22.alpha) |
| Firmware | 配套 CANN 7.0.0.beta1 | {device}-npu-firmware_{version}.run | [link](https://www.hiascend.com/hardware/firmware-drivers/community?product=2&model=2&cann=7.0.0.beta1&driver=1.0.22.alpha) |
| CANN nnae | 7.0.0.beta1 | Ascend-cann-nnae_7.0.0_linux-{arch}.run | [link](https://www.hiascend.com/developer/download/community/result?module=cann&cann=7.0.0.beta1) |
| CANN kernels(可选) | 7.0.0.beta1 | Ascend-cann-kernels-{device}_7.0.0_linux.run | [link](https://www.hiascend.com/developer/download/community/result?module=cann&cann=7.0.0.beta1) |

#### 安装

```shell
# 注意：新机器安装先driver后firmware，驱动及固件升级场景，先firmware后driver
bash {device}-npu-driver_{version}_linux-{arch}.run --full
bash {device}-npu-firmware_{version}.run --full
bash Ascend-cann-nnae_7.0.0_linux-{arch}.run --install
bash Ascend-cann-kernels-{device}_7.0.0_linux.run --install

pip uninstall te topi hccl -y
pip install sympy
pip install /usr/local/Ascend/nnae/latest/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/nnae/latest/lib64/hccl-*-py3-none-any.whl
reboot   # 重启
```

#### 配置环境变量

```shell
source /usr/local/Ascend/nnae/set_env.sh
```

### 3. MindSpore 安装

```shell
pip install mindspore==2.2.14

# 查看版本号，离线推理MindSpore仅使用CPU
python -c "import mindspore;mindspore.set_context(device_target='CPU');mindspore.run_check()"
```

### 4. MindSpore Lite 安装

| software | version | package name | download |
| --- | --- | --- | --- |
| 推理工具包 | 2.2.14 | mindspore-lite-2.2.14-linux-{arch}.tar.gz | [link](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html#2-2-14) |
| Python 接口 Wheel安装包 | 2.2.14 | mindspore_lite-2.2.14-{python_version}-linux_{arch}.whl | [link](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html#2-2-14) |

推理工具包安装时直接解压即可，并注意设置环境变量：

```shell
tar -xvf mindspore-lite-2.2.14-linux-{arch}.tar.gz
cd mindspore-lite-2.2.14-linux-{arch}/
export LITE_HOME=${PWD}    # 推理工具tar包解压后实际路径
export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/runtime/third_party/dnnl:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
export PATH=$LITE_HOME/tools/converter/converter:$LITE_HOME/tools/benchmark:$PATH
```

如果使用python接口，使用pip安装所需的whl包

```shell
pip install mindspore_lite-2.2.14-{python_version}-linux_{arch}.whl
```

### 5. 安装 MindOCR

```shell
git clone https://github.com/mindspore-lab/mindocr.git
cd mindocr
pip install -e .
```

> 使用 `-e` 代表可编辑模式，可以帮助解决潜在的模块导入问题。
