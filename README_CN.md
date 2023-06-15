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
[教程](#教程) |
[模型列表](#模型列表) |
[OCR数据集](#数据集) |
[重要信息](#重要信息)

</div>

## 概述
MindOCR是一个基于[MindSpore](https://www.mindspore.cn/en)框架的OCR开发及应用的开源工具箱，可以帮助用户训练、应用业界最有优的文本检测、文本识别模型，例如DBNet/DBNet++和CRNN/SVTR，以实现图像文本理解的需求。


<details open>
<summary> 主要特性 </summary>

- **模块化设计**: MindOCR将OCR任务解耦成多个可配置模块，用户只需修改几行代码，就可以轻松地在定制化的数据和模型上配置训练、评估的全流程；
- **高性能**: MindOCR提供的预训练权重和训练方法可以使其达到OCR任务上具有竞争力的表现；
- **易用性**: MindOCR提供易用工具帮助在真实世界数据中进行文本的检测和识别。
</details>


## 安装

### 依赖

请运行如下代码安装依赖包：
```shell
pip install -r requirements.txt
```

此外，请按[官方指引](https://www.mindspore.cn/install)安装MindSpore(>=1.9) 来适配您的机器。如果需要在分布式模式下进行训练，还请安装[openmpi](https://www.open-mpi.org/software/ompi/v4.0/)。

#### 预备环境

| 环境         | 版本   |
|-------------|-------|
| MindSpore   | >=1.9 |
| Python      | >=3.7 |

MindOCR基于MindSpore框架构建，支持包含CPU/GPU/NPU的多种硬件设备。MindOCR可适配以下框架版本，更多信息请参见安装链接。

- mindspore >= 1.9  [[安装](https://www.mindspore.cn/install)]
- python >= 3.7
- openmpi 4.0.3 (for distributed training/evaluation)  [[安装](https://www.open-mpi.org/software/ompi/v4.0/)]
- mindspore lite (for inference)  [[安装](docs/en/inference/environment_en.md)]

#### 依赖

```shell
pip install -r requirements.txt
```
**小建议:**

如果无法导入sckit_image，请设置环境变量`$LD_PRELOAD`，如下所示([相关opencv issue](https://github.com/opencv/opencv/issues/14884))：

    ```shell
    export LD_PRELOAD=path/to/scikit_image.libs/libgomp-d22c30c5.so.1.0.0:$LD_PRELOAD
    ```

#### 通过源文件安装（推荐）

```shell
git clone https://github.com/mindspore-lab/mindocr.git
cd mindocr
pip install -e .
```
> 使用 `-e` 代表可编辑模式，可以帮助解决潜在的模块导入问题。

#### 通过PyPI安装

```shell
pip install mindocr
```
>由于此项目正在积极开发中，从PyPI安装的版本目前已过期，我们将很快更新，敬请期待。

## 快速上手

有MindOCR的帮助，在图片上检测文本框位置、识别文字内容变得很简单。
```shell
python tools/infer/text/predict_system.py --image_dir {path_to_img or dir_to_imgs} \
                                          --det_algorithm DB++  \
                                          --rec_algorithm CRNN
```

训练之后，结果将被默认保存在`./inference_results`路径，下面是一个结果样例：
<p align="center">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/c1f53970-8618-4039-994f-9f6dc1eee1dd" width=600 />
</p>
<p align="center">
  <em> 文本检测、识别结果可视化 </em>
</p>

更多使用方法，请参考[教程](#教程)

### 训练与评估快速指南

使用`tools/t.py`脚本可以很容易地训练OCR模型，该脚本支持文本检测和识别训练。
```shell
python tools/train.py --config {path/to/model_config.yaml}
```
`--config` 参数用于指定yaml文件的路径，该文件定义要训练的模型和训练策略，包括数据处理流程、优化器、学习率调度器等。

MindOCR在`configs`文件夹中提供最新最优的OCR模型及训练策略，用户可以将其适配到自己的任务或数据集上。例如，运行如下代码加载预置的DBNet++和CRNN模型及训练策略。
```shell
# train text detection model DBNet++ on icdar15 dataset
python tools/train.py --config configs/det/dbnet/db++_r50_icdar15.yaml
```
```shell
# train text recognition model CRNN on icdar15 dataset
python tools/train.py --config configs/rec/crnn/crnn_icdar15.yaml
```

类似的，使用`tools/eval.py` 脚本可以很容易的评估已经训练好的模型，如下所示：
```shell
python tools/eval.py \
    --config {path/to/model_config.yaml} \
    --opt eval.dataset_root={path/to/your_dataset} eval.ckpt_load_path={path/to/ckpt_file}
```

更多使用方法，请参考[教程](#教程)中的模型训练章节。

## 教程

- 数据集
    - [数据集准备](tools/dataset_converters/README.md)
    - [数据增强策略](docs/en/tutorials/transform_tutorial.md)
- 模型训练
    - [Yaml配置文件]() // 敬请期待
    - [文本检测]()  // 敬请期待
    - [文本识别](docs/en/tutorials/training_recognition_custom_dataset.md)
    - [使用 OpenMPI/HCCL在昇腾服务器上分布式训练](docs/cn/tutorials/distribute_train_CN.md)
    - [高级：梯度累积，EMA，断点续训等](docs/en/tutorials/advanced_train.md)
- 推理与部署
    - [在昇腾310服务器上使用Python/C++推理](docs/en/inference/inference_tutorial_en.md)
    - [Python在线推理](tools/infer/text/README.md)
- 开发者指南
    - [定制数据集](mindocr/data/README.md)
    - [定制数据增强方法](mindocr/data/transforms/README.md)
    - [定制新模型](mindocr/models/README.md)
    - [定制后处理方法](mindocr/postprocess/README.md)

## 模型列表

<details open>
<summary>文本检测</summary>
    
- [x] [DBNet](configs/det/dbnet/README.md) (AAAI'2020)
- [x] [DBNet++](configs/det/dbnet/README.md) (TPAMI'2022)
- [x] [PSENet](configs/det/psenet/README.md) (CVPR'2019)
- [x] [EAST](configs/det/east/README.md)(CVPR'2017)
- [ ] [FCENet](https://arxiv.org/abs/2104.10442) (CVPR'2021) [敬请期待]
</details>

<details open>
    
<summary>文本识别</summary>
    
- [x] [CRNN](configs/rec/crnn/README.md) (TPAMI'2016)
- [x] [CRNN-Seq2Seq/RARE](configs/rec/rare/README.md) (CVPR'2016)
- [x] [SVTR](configs/rec/svtr/README.md) (IJCAI'2022)
- [ ] [ABINet](https://arxiv.org/abs/2103.06495) (CVPR'2021) [coming soon]
</details>

请参见[configs](./configs).查看以上模型的详细精度性能结果。

[MindSpore Lite](https://www.mindspore.cn/lite)和[ACL](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/aclcppdevg/aclcppdevg_000004.html)模型推理的支持列表，请参见[MindOCR支持模型列表](docs/en/inference/models_list_en.md) and [第三方模型推理支持列表](docs/en/inference/models_list_thirdparty_en.md)。

## 数据集

<details open>
<summary>通用OCR数据集</summary>
    
- [x] ICDAR2015 [论文](https://rrc.cvc.uab.es/files/short_rrc_2015.pdf) [主页](https://rrc.cvc.uab.es/?ch=4) [下载说明](docs/en/datasets/icdar2015.md)
- [x] Total-Text [论文](https://arxiv.org/abs/1710.10400) [主页](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Dataset) [下载说明](docs/en/datasets/totaltext.md)
- [x] Syntext150k [论文](https://arxiv.org/abs/2002.10200) [主页](https://github.com/aim-uofa/AdelaiDet) [下载说明](docs/en/datasets/syntext150k.md)
- [x] MLT2017 [paper](https://ieeexplore.ieee.org/abstract/document/8270168) [主页](https://rrc.cvc.uab.es/?ch=8&com=introduction) [下载说明](docs/en/datasets/mlt2017.md)
- [x] MSRA-TD500 [论文](https://ieeexplore.ieee.org/abstract/document/6247787) [主页](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500)) [下载说明](docs/en/datasets/td500.md)
- [x] SCUT-CTW1500 [论文](https://www.sciencedirect.com/science/article/pii/S0031320319300664) [主页](https://github.com/Yuliang-Liu/Curve-Text-Detector) [下载说明](docs/en/datasets/ctw1500.md)
</details>

正在持续更新中。

## 重要信息

### 更新

- 2023/06/07
1. 增加新模型
    - 文本检测[PSENet](configs/det/psenet)
    - 文本检测[EAST](configs/det/east)
    - 文本识别[SVTR](configs/rec/svtr)
2. 添加更多基准数据集及其结果
    - [totaltext](docs/cn/datasets/totaltext_CN.md)
    - [mlt2017](docs/cn/datasets/mlt2017_CN.md)
    - [chinese_text_recognition](docs/cn/datasets/chinese_text_recognition_CN.md)
3. 增加断点重训(resume training)功能，可在训练意外中断时使用。如需使用，请在配置文件中`model`字段下增加`resume`参数，允许传入具体路径`resume: /path/to/train_resume.ckpt`或者通过设置`resume: True`来加载在ckpt_save_dir下保存的trian_resume.ckpt
4. 改进检测模块的后处理部分：默认情况下，将检测到的文本多边形重新缩放到原始图像空间，可以通过在`eval.dataset.output_columns`列表中增加"shape_list"实现。
5. 重构在线推理以支持更多模型，详情请参见[README.md](tools/infer/text/README.md) 。

- 2023/05/15
1. 增加新模型
    - 文本检测[DBNet++](configs/det/dbnet)
    - 文本识别[CRNN-Seq2Seq](configs/rec/rare)
    - 在SynthText数据集上预训练的[DBNet](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_synthtext-40655acb.ckpt)
2. 添加更多基准数据集及其结果
    - [SynthText](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c), [MSRA-TD500](docs/cn/datasets/td500_CN.md), [CTW1500](docs/cn/datasets/ctw1500_CN.md)
    - DBNet的更多基准结果可以[在此找到](configs/det/dbnet/README_CN.md).
3. 添加用于保存前k个checkpoint的checkpoint manager并改进日志。
4. Python推理代码重构。
5. Bug修复：对大型数据集使用平均损失meter，在AMP训练中对ctcloss禁用`pred_cast_fp32`，修复存在无效多边形的错误。

- 2023/05/04
1. 支持加载自定义的预训练checkpoint， 通过在yaml配置中将`model-pretrained`设置为checkpoint url或本地路径来使用。
2. 支持设置执行包括旋转和翻转在内的数据增强操作的概率。
3. 为模型训练添加EMA功能，可以通过在yaml配置中设置`train-ema`（默认值：False）和`train-ema_decay`来启用。
4. 参数修改：`num_columns_to_net` -> `net_input_column_index`: 输入网络的columns数量改为输入网络的columns索引
5. 参数修改：`num_columns_of_labels` -> `label_column_index`: 用索引替换数量，以表示lebel的位置。

- 2023/04/21
1. 添加参数分组以支持训练中的正则化。用法：在yaml config中添加`grouping_strategy`参数以选择预定义的分组策略，或使用`no_weight_decay_params`参数选择要从权重衰减中排除的层（例如，bias、norm）。示例可参考`configs/rec/crn/crnn_icdar15.yaml`
2. 添加梯度积累，支持大批量训练。用法：在yaml配置中添加`gradient_accumulation_steps`，全局批量大小=batch_size * devices * gradient_aaccumulation_steps。示例可参考`configs/rec/crn/crnn_icdar15.yaml`
3. 添加梯度裁剪，支持训练稳定。通过在yaml配置中将`grad_clip`设置为True来启用。

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
