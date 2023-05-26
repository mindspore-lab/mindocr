[English](../../en/inference/environment_en.md) | 中文

## 推理 - 运行环境安装

MindOCR支持Ascend310/Ascend310P设备的推理。

请确保系统正确安装了昇腾AI处理器配套软件包，如果没有安装，请先参考[安装昇腾AI处理器配套软件包](https://www.mindspore.cn/install#安装昇腾ai处理器配套软件包)小节进行安装。

MindOCR后端支持[ACL](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/aclcppdevg/aclcppdevg_000004.html)和[MindSpore Lite](https://www.mindspore.cn/lite/docs/zh-CN/master/index.html)两种推理模式，请根据使用场景选择安装环境，区别如下：

|        | ACL                                                                                                                     | Lite                                                                                                  |
|:-------|:------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------|
| 文件后缀 | .om                                                                                                                     | .mindir                                                                                               |
| 转换工具 | [ATC](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/atctool/atctool_000001.html) | [converter_lite](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/converter_tool.html) |
| 适用场景 | 第三方模型的推理，模型需要转为OM文件                                                                                          | 1. MindSpore模型的推理<br>2. 第三方模型的推理<br>模型需要转为MindIR文件                                       |

### 1. ACL推理

对于MindOCR的ACL方式推理，目前Python侧依赖于[MindX](https://www.hiascend.com/software/Mindx-sdk)的Python
API接口，该接口暂只支持Python3.9。

| 环境    | 版本  |
|:-------|:------|
| Python | 3.9   |
| MindX  | 3.0.0 |

在Python3.9环境基础上，下载[MindX](https://www.hiascend.com/zh/software/mindx-sdk/commercial)的mxVision
SDK安装包，参考[指导教程](https://www.hiascend.com/document/detail/zh/mind-sdk/300/quickstart/visionquickstart/visionquickstart_0003.html)进行安装，主要步骤如下：

```shell
# 增加可执行权限
chmod +x Ascend-mindxsdk-mxvision_{version}_linux-{arch}.run
# 执行安装命令，如果提示需指定cann包路径，则增加参数如:--cann-path=/usr/local/Ascend/latest
./Ascend-mindxsdk-mxvision_{version}_linux-{arch}.run --install
# 设置环境变量
source mxVision/set_env.sh
```

安装完毕之后测试一下mindx是否可以正常导入：`python -c "import mindx"`

如果提示找不到mindx，则转到mxVision/python目录下，安装对应的whl包：

```
cd mxVision/python
pip install *.whl
```

### 2. MindSpore Lite推理

对于MindOCR的MindSpore Lite推理，需要安装2.0.0-rc1以上版本的[MindSpore Lite](https://www.mindspore.cn/lite/docs/zh-CN/master/index.html)的**云侧**推理工具包。

先[下载](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)Ascend版的云侧版本的推理工具包tar.gz文件，以及Python接口Wheel包。

下载地址中提供了3.7版本的Python包，如需其它版本可参考[编译](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/build.html)教程。

推理工具包安装时直接解压即可，并注意设置环境变量：

```shell
export LITE_HOME=/your_path_to/mindspore-lite
export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
export PATH=$LITE_HOME/tools/converter/converter:$LITE_HOME/tools/benchmark:$PATH
```

Python接口的Wheel包则使用pip安装：

```shell
pip install mindspore_lite-{version}-{python_version}-linux_{arch}.whl
```
