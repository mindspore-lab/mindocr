<div align="center">

# MindOCR

[![CI](https://github.com/mindspore-lab/mindocr/actions/workflows/ci.yml/badge.svg)](https://github.com/mindspore-lab/mindocr/actions/workflows/ci.yml)
[![license](https://img.shields.io/github/license/mindspore-lab/mindocr.svg)](https://github.com/mindspore-lab/mindocr/blob/main/LICENSE)
[![open issues](https://img.shields.io/github/issues/mindspore-lab/mindocr)](https://github.com/mindspore-lab/mindocr/issues)
[![PRs](https://img.shields.io/badge/PRs-welcome-pink.svg)](https://github.com/mindspore-lab/mindocr/pulls)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


[English](README.md) | 中文

[概述](#概述) |
[安装](#安装) |
[快速上手](#快速上手) |
[模型列表](#模型列表) |
[重要信息](#重要信息)

</div>

## 概述
MindOCR是一个基于[MindSpore](https://www.mindspore.cn/en)框架的OCR开发及应用的开源工具箱，可以帮助用户训练、应用业界最有优的文本检测、文本识别模型，例如DBNet/DBNet++和CRNN/SVTR，以实现图像文本理解的需求。


<details open>
<summary> 主要特性 </summary>

- **模块化设计**: MindOCR将OCR任务解耦成多个可配置模块，用户只需修改几行代码，就可以轻松地在定制化的数据和模型上配置训练、评估的全流程；
- **高性能**: MindOCR提供的预训练权重和训练方法可以使其达到OCR任务上具有竞争力的表现；
- **易用性**: MindOCR提供易用工具帮助在真实世界数据中进行文本的检测和识别（敬请期待）。
</details>


## 安装

### 依赖

请运行如下代码安装依赖包：
```shell
pip install -r requirements.txt
```

此外，请按[官方指引](https://www.mindspore.cn/install)安装MindSpore(>=1.9) 来适配您的机器。如果需要在分布式模式下进行训练，还请安装[openmpi](https://www.open-mpi.org/software/ompi/v4.0/)。

| 环境         | 版本   |
|-------------|-------|
| MindSpore   | >=1.9 |
| Python      | >=3.7 |


> 注意：
> - 如果[使用ACL推理](#21-使用mindspore-lite和acl推理)，Python版本需为3.9。
> - 如果遇到scikit_image导入错误，参考[此处](https://github.com/opencv/opencv/issues/14884)，你需要设置环境变量`$LD_PRELOAD`，命令如下。替换`path/to`为你的目录。
>   ```shell
>   export LD_PRELOAD=path/to/scikit_image.libs/libgomp-d22c30c5.so.1.0.0:$LD_PRELOAD
>   ```

### 通过PyPI安装

敬请期待

### 通过源文件安装

最新版的MindOCR可以通过如下命令安装：
```shell
pip install git+https://github.com/mindspore-lab/mindocr.git
```

> 注意：MindOCR目前暂时只在MindSpore>=1.9版本，Linux系统，GPU/Ascend设备上进行过测试。

## 快速上手

### 1. 模型训练评估

#### 1.1 文本检测

MindOCR支持多种文本检测模型及数据集，在此我们使用**DBNet**模型和**ICDAR2015**数据集进行演示。请参考[DBNet模型文档](configs/det/dbnet/README_CN.md)。


### 1.2 文本识别

MindOCR支持多种文本识别模型及数据集，在此我们使用**CRNN**模型和**LMDB**数据集进行演示。请参考[CRNN模型文档](configs/rec/crnn/README_CN.md)。


### 2. 推理与部署

#### 2.1 使用MindSpore Lite和ACL推理(Ascend 310)

MindOCR集成了[MindSpore Lite](https://www.mindspore.cn/lite)和[ACL](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/aclcppdevg/aclcppdevg_000004.html)推理后端，
集成了文本检测、分类和识别串联推理。

具体说明请参考[MindOCR 310推理](docs/cn/inference/inference_tutorial_cn.md)。

#### 2.2 使用原生MindSpore在线推理(CPU/GPU/Ascend 910)

MindOCR提供易用的文本检测识别推理工具，支持CPU/GPU/Ascend 910硬件平台。在线推理基于使用MindOCR训练完成的模型进行推理。

具体用法和效果请参考 [MindOCR在线推理](tools/infer/text/README.md)。

## 模型列表

<details open>
<summary>文本检测</summary>

- [x] [DBNet](configs/det/dbnet/README.md) (AAAI'2020) 
- [x] [DBNet++](configs/det/dbnet/README.md) (TPAMI'2022)
- [x] [PSENet](configs/det/psenet/README.md) (CVPR'2019)
- [x] [EAST](configs/det/east/README.md)(CVPR'2017)
- [ ] [FCENet](https://arxiv.org/abs/2104.10442) (CVPR'2021) [coming soon]

</details>

<details open>
<summary>文本识别</summary>


- [x] [CRNN](configs/rec/crnn/README.md) (TPAMI'2016)
- [x] [CRNN-Seq2Seq/RARE](configs/rec/rare/README.md) (CVPR'2016)
- [x] [SVTR](configs/rec/svtr/README.md) (IJCAI'2022) 
- [ ] [ABINet](https://arxiv.org/abs/2103.06495) (CVPR'2021) [coming soon]


模型训练的配置及性能结果请见[configs](./configs).

MindSpore Lite和ACL模型推理的支持列表，请见[MindOCR模型推理支持列表](docs/cn/inference/models_list_cn.md)和[第三方模型推理支持列表](docs/cn/inference/models_list_thirdparty_cn.md).

## 数据集
### 下载

我们提供以下数据集的下载说明。

<details open>
<summary>文本检测</summary>

- [x] ICDAR2015 [论文]((https://rrc.cvc.uab.es/files/short_rrc_2015.pdf)) [主页]((https://rrc.cvc.uab.es/?ch=4)) [下载说明](docs/cn/datasets/icdar2015_CN.md)

- [x] Total-Text [论文](https://arxiv.org/abs/1710.10400) [主页](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Dataset) [下载说明](docs/cn/datasets/totaltext_CN.md)

- [x] Syntext150k [论文](https://arxiv.org/abs/2002.10200) [主页](https://github.com/aim-uofa/AdelaiDet) [下载说明](docs/cn/datasets/syntext150k_CN.md)

- [x] MLT2017 [论文](https://ieeexplore.ieee.org/abstract/document/8270168) [主页](https://rrc.cvc.uab.es/?ch=8&com=introduction) [下载说明](docs/cn/datasets/mlt2017_CN.md)

- [x] MSRA-TD500 [论文](https://ieeexplore.ieee.org/abstract/document/6247787) [主页](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500)) [下载说明](docs/cn/datasets/td500_CN.md)

- [x] SCUT-CTW1500 [论文](https://www.sciencedirect.com/science/article/pii/S0031320319300664) [主页](https://github.com/Yuliang-Liu/Curve-Text-Detector) [下载说明](docs/cn/datasets/ctw1500_CN.md)

</details>

### 转换

在 `DATASETS_DIR` 文件夹中下载这些数据集后，您可以运行 `bash tools/convert_datasets.sh` 将所有下载的数据集转换为目标格式。[这里](tools/dataset_converters/README_CN.md)有一个 icdar2015 数据集转换的例子。


## 重要信息

### 变更日志
- 2023/05/04
1. 参数修改：`num_columns_to_net` -> `net_input_column_index`: 输入网络的columns数量改为输入网络的columns索引
2. 参数修改：`num_columns_of_labels` -> `label_column_index`: 代表label的columns数量改为代表label的columns索引

- 2023/03/23
1. 增加dynamic loss scaler支持, 且与drop overflow update兼容。如需使用, 请在配置文件中增加`loss_scale`字段并将`type`参数设为`dynamic`，参考例子请见`configs/rec/crnn/crnn_icdar15.yaml`


- 2023/03/20
1. 参数名修改：`output_keys` -> `output_columns`；`num_keys_to_net` -> `num_columns_to_net`；
2. 更新数据流程。

- 2023/03/13
1. 增加系统测试和CI工作流；
2. 增加modelarts平台适配器，使得支持在OpenI平台上训练，在OpenI平台上训练需要以下步骤：
  ```text
    i)   在OpenI云平台上创建一个训练任务；
    ii)  在网页上关联数据集，如ic15_mindocr；
    iii) 增加 `config` 参数，在网页的UI界面配置yaml文件路径，如'/home/work/user-job-dir/V0001/configs/rec/test.yaml'；
    iv)  在网页的UI界面增加运行参数`enable_modelarts`并将其设置为True；
    v)   填写其他项并启动训练任务。
  ```

- 2023/03/08
1. 增加评估脚本 with  arg `ckpt_load_path`；
2. Yaml文件中的`ckpt_save_dir`参数从`system` 移动到 `train`；
3. 增加rop_overflow_update控制。

### 如何贡献

我们欢迎包括问题单和PR在内的所有贡献，来让MindOCR变得更好。

请参考[CONTRIBUTING.md](CONTRIBUTING.md)作为贡献指南，请按照[Model Template and Guideline](mindocr/models/README.md)的指引贡献一个适配所有接口的模型，多谢合作。

### 许可

本项目遵从[Apache License 2.0](LICENSE)开源许可。

### 引用

如果本项目对您的研究有帮助，请考虑引用：

```latex
@misc{MindSpore OCR 2023,
    title={{MindSpore OCR }:MindSpore OCR Toolbox},
    author={MindSpore Team},
    howpublished = {\url{https://github.com/mindspore-lab/mindocr/}},
    year={2023}
}
```
