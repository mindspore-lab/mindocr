
# MindOCR

<!--
[![license](https://img.shields.io/github/license/mindspore-lab/mindocr.svg)](https://github.com/mindspore-lab/mindocr/blob/main/LICENSE)
[![open issues](https://img.shields.io/github/issues/mindspore-lab/mindocr)](https://github.com/mindspore-lab/mindocr/issues)
[![PRs](https://img.shields.io/badge/PRs-welcome-pink.svg)](https://github.com/mindspore-lab/mindocr/pulls)
 -->
[English](README.md) | 中文

[概述](#introduction) |
[安装](#installation) |
[快速上手](#quick-start) |
[模型列表](#supported-models-and-performance) |
[注释](#notes)


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

此外，请按[官方指引](https://www.mindspore.cn/install)安装MindSpore(>=1.8.1) 来适配您的机器。如果需要在分布式模式下进行训练，还请安装[openmpi](https://www.open-mpi.org/software/ompi/v4.0/)。


### 通过PyPI安装

敬请期待

### 通过源文件安装

最新版的MindOCR可以通过如下命令安装：
```shell
pip install git+https://github.com/mindspore-lab/mindocr.git
```

> 注意：MindOCR目前暂时只在MindSpore>=1.8.1版本，Linux系统，GPU/Ascend设备上进行过测试。

## 快速上手

### 训练文本检测模型

MindOCR支持多种文本检测模型及数据集，在此我们使用**DBNet** 模型和 **ICDAR2015**数据集进行演示。 

#### 1. 数据准备

请从[该网址](https://rrc.cvc.uab.es/?ch=4&com=downloads)下载ICDAR2015数据集，然后参考[数据转换](tools/dataset_converters/README_CN.md)对数据集标注进行格式化。

完成数据准备工作后，数据的目录结构应该如下所示： 

``` text
.
├── test
│   ├── images
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   └── ...
│   └── det_gt.txt
└── train
    ├── images
    │   ├── img_1.jpg
    │   ├── img_2.jpg
    │   └── ....jpg
    └── det_gt.txt
```

#### 2. 配置Yaml文件

在`configs/det`中选择一个包含目标预训练模型和数据流程的yaml配置文件，这里我们选择`configs/det/dbnet/db_r50_icdar15.yaml`。

然后，按照以下指引更改数据配置参数：
``` yaml
train:
  dataset:
    data_dir: PATH/TO/TRAIN_IMAGES_DIR
    label_file: PATH/TO/TRAIN_LABELS.txt
eval:
  dataset:
    data_dir: PATH/TO/TEST_IMAGES_DIR
    label_file: PATH/TO/TEST_LABELS.txt
```

【可选】可以根据CPU核的数量设置`num_workers`参数的值；如果需要在分布式模式下训练，可修改`distribute`为True。

#### 3. 训练

运行以下命令开始模型训练：

``` shell 
# train dbnet on ic15 dataset
python tools/train.py --config configs/det/dbnet/db_r50_icdar15.yaml
```

如果在分布式模式下，请运行命令：

```shell
# n is the number of GPUs/NPUs
mpirun --allow-run-as-root -n 2 python tools/train.py --config configs/det/dbnet/db_r50_icdar15.yaml
```
> 注意：请确保yaml文件中的`distribute`参数为True。


训练结果 (包括checkpoint、每个epoch的性能和曲线图)将被保存在yaml配置文件的`ckpt_save_dir`参数配置的路径下，默认为 "./tmp_det/"。 

#### 4. 评估

评估环节，在yaml配置文件中将`ckpt_load_path`参数配置为checkpoint文件的路径，然后运行： 

``` shell
python tools/eval.py --config configs/det/dbnet/db_r50_icdar15.yaml
```


### 训练文本识别模型

MindOCR支持多种文本识别模型及数据集，在此我们使用**CRNN** 模型和 **LMDB**数据集进行演示。

#### 1. 数据准备

参考[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here)，从[这里](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0)下载LMDB数据集。

一共有如下.zip压缩数据文件：
- `data_lmdb_release.zip` 包含训练、验证及测试的全部数据；
- `validation.zip` 是验证数据集的合集；
- `evaluation.zip` 包含多个评估数据集。 

解压文件并完成数据准备操作后，数据文件夹结构如下：

``` text
.
├── train
│   ├── MJ
│   │   ├── data.mdb
│   │   ├── lock.mdb
│   ├── ST
│   │   ├── data.mdb
│   │   ├── lock.mdb
└── validation
|   ├── data.mdb
|   ├── lock.mdb
└── evaluation
    ├── IC03
    │   ├── data.mdb
    │   ├── lock.mdb
    ├── IC13
    │   ├── data.mdb
    │   ├── lock.mdb
    └── ...
```

#### 2. 配置Yaml文件

在`configs/rec`中选择一个包含目标预训练模型和数据流程的yaml配置文件，这里我们选择`configs/rec/crnn/crnn_resnet34.yaml`。

相应的更改数据配置参数：
``` yaml
train:
  dataset:
    type: LMDBDataset
    data_dir: lmdb_data/rec/train/
eval:
  dataset:
    type: LMDBDataset
    data_dir: lmdb_data/rec/validation/
```
【可选】可以根据CPU核的数量设置`num_workers`参数的值；如果需要在分布式模式下训练，可修改`distribute`为True。

#### 3. 训练

运行以下命令开始模型训练： 

``` shell 
# train crnn on MJ+ST dataset
python tools/train.py --config configs/rec/crnn/crnn_resnet34.yaml
```

如果在分布式模式下，请运行命令：

```shell
# n is the number of GPUs/NPUs
mpirun --allow-run-as-root -n 2 python tools/train.py --config configs/rec/crnn/crnn_resnet34.yaml
```
> 注意：请确保yaml文件中的`distribute`参数为True。


训练结果 (包括checkpoint、每个epoch的性能和曲线图)将被保存在yaml配置文件的`ckpt_save_dir`参数配置的路径下，默认为 "./tmp_rec/"。 

#### 4. 评估

评估环节，在yaml配置文件中将`ckpt_load_path`参数配置为checkpoint文件的路径，然后运行： 

``` shell
python tools/eval.py --config configs/rec/crnn/crnn_resnet34.yaml
```

### 推理与部署

#### 使用MX Engine推理

请参考[mx_infer](docs/cn/inference_cn.md)

#### 使用Lite推理 

敬请期待

#### 使用原生MindSpore推理

敬请期待

## 支持模型及性能

### 文本检测  

下表是目前支持的文本检测模型和它们在ICDAR2015测试数据集上的精度数据：

| **模型**  | **骨干网络**  | **预训练**      | **Recall** | **Precision** | **F-score** | **配置文件**                                            | 
|-----------|--------------|----------------|------------|---------------|-------------|-----------------------------------------------------|
| DBNet     | ResNet-50    | ImageNet       | 81.97%     | 86.05%        | 83.96%      | [YAML](configs/det/dbnet/dbnet/db_r50_icdar15.yaml) | 
| DBNet++   | ResNet-50    | ImageNet       | 82.02%     | 87.38%        | 84.62%      | [YAML](configs/det/dbnet++/db++_r50_icdar15.yaml)   |

### 文本识别

下表是目前支持的文本识别模型和它们在公开测评数据集 (IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE) 上的精度数据：


| **模型** | **骨干网络** | **平均准确率**| **配置文件** | 
|-----------|--------------|----------------|------------|
| CRNN     | VGG7        | 82.03% 	| [YAML](configs/rec/crnn/crnn_vgg7.yaml)    | 
| CRNN     | Resnet34_vd    | 84.45% 	| [YAML](configs/rec/crnn/crnn_resnet34.yaml)     |


## 注释

### 变更日志
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

本项目遵从[Apache License 2.0](LICENSE.md)开源许可。

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
