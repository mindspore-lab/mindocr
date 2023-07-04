---
hide:
  - navigation
---

<div align="center" markdown>

# MindOCR

[![CI](https://github.com/mindspore-lab/mindocr/actions/workflows/ci.yml/badge.svg)](https://github.com/mindspore-lab/mindocr/actions/workflows/ci.yml)
[![license](https://img.shields.io/github/license/mindspore-lab/mindocr.svg)](https://github.com/mindspore-lab/mindocr/blob/main/LICENSE)
[![open issues](https://img.shields.io/github/issues/mindspore-lab/mindocr)](https://github.com/mindspore-lab/mindocr/issues)
[![PRs](https://img.shields.io/badge/PRs-welcome-pink.svg)](https://github.com/mindspore-lab/mindocr/pulls)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

## 简介
MindOCR是一个基于[MindSpore](https://www.mindspore.cn/en) 框架开发的OCR开源工具箱，集成系列主流文字检测识别的算法、模型，并提供易用的训练和推理工具，可以帮助用户快速开发和应用业界SoTA文本检测、文本识别模型，如DBNet/DBNet++和CRNN/SVTR，满足图像文档理解的需求。


<details open markdown>
<summary> 主要特性 </summary>

- **模块化设计**: MindOCR将OCR任务解耦成多个可配置模块，用户只需修改几行代码，就可以轻松地在定制化的数据和模型上配置训练、评估的全流程；
- **高性能**: MindOCR提供的预训练权重和训练方法可以使其达到OCR任务上具有竞争力的表现；
- **易用性**: MindOCR提供易用工具帮助在真实世界数据中进行文本的检测和识别。
</details>


## 安装教程

#### MindSpore相关环境准备

MindOCR基于MindSpore AI框架（支持CPU/GPU/NPU）开发，并适配以下框架版本。安装方式请参见下方的安装链接。

- mindspore >= 1.9  [[安装](https://www.mindspore.cn/install)]
- python >= 3.7
- openmpi 4.0.3 (for distributed training/evaluation)  [[安装](https://www.open-mpi.org/software/ompi/v4.0/)]
- mindspore lite (for inference)  [[安装](inference/environment.md)]

#### 包依赖

```shell
pip install -r requirements.txt
```
**提示:**

- 如果无法导入sckit_image，请设置环境变量`$LD_PRELOAD`，如下所示([相关opencv issue](https://github.com/opencv/opencv/issues/14884))：

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

## 快速开始

### 文字检测和识别示例

安装完MindOCR后，我们就很方便地进行任意图像的文本检测和识别，如下。

```shell
python tools/infer/text/predict_system.py --image_dir {path_to_img or dir_to_imgs} \
                                          --det_algorithm DB++  \
                                          --rec_algorithm CRNN
```

运行结束后，结果将被默认保存在`./inference_results`路径，可视化结果如下：
<p align="center">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/c1f53970-8618-4039-994f-9f6dc1eee1dd" width=600 />
</p>
<p align="center">
  <em> 文本检测、识别结果可视化 </em>
</p>

可以看到图像中的文字块均被检测出来并正确识别。更详细的用法介绍，请参考推理[教程](#_7)。

### 模型训练与评估-快速指南

使用`tools/train.py`脚本可以很容易地训练OCR模型，该脚本可支持文本检测和识别模型训练。
```shell
python tools/train.py --config {path/to/model_config.yaml}
```
`--config` 参数用于指定yaml文件的路径，该文件定义要训练的模型和训练策略，包括数据处理流程、优化器、学习率调度器等。

MindOCR在`configs`文件夹中提供系列SoTA的OCR模型及其训练策略，用户可以快速将其适配到自己的任务或数据集上，参考例子如下

```shell
# train text detection model DBNet++ on icdar15 dataset
python tools/train.py --config configs/det/dbnet/db++_r50_icdar15.yaml
```
```shell
# train text recognition model CRNN on icdar15 dataset
python tools/train.py --config configs/rec/crnn/crnn_icdar15.yaml
```

类似的，使用`tools/eval.py` 脚本可以很容易地评估已训练好的模型，如下所示：
```shell
python tools/eval.py \
    --config {path/to/model_config.yaml} \
    --opt eval.dataset_root={path/to/your_dataset} eval.ckpt_load_path={path/to/ckpt_file}
```

更多使用方法，请参考[使用教程](#_7)中的模型训练章节。

## 使用教程

- 数据集
    - [数据集准备](datasets/converters.md)
    - [数据增强策略](tutorials/transform_tutorial.md)
- 模型训练
    - [Yaml配置文件](tutorials/yaml_configuration.md)
    - [文本检测](tutorials/training_detection_custom_dataset.md)
    - [文本识别](tutorials/training_recognition_custom_dataset.md)
    - [分布式训练](tutorials/distribute_train.md)
    - [进阶技巧：梯度累积，EMA，断点续训等](tutorials/advanced_train.md)
- 推理与部署
    - [基于Python/C++和昇腾310的OCR推理](inference/inference_tutorial.md)
    - [基于Python的OCR在线推理](mkdocs/online_inference.md)
- 开发者指南
    - [如何自定义数据集](mkdocs/customize_dataset.md)
    - [如何自定义数据增强方法](mkdocs/customize_data_transform.md)
    - [如何创建新的OCR模型](mkdocs/customize_model.md)
    - [如何自定义后处理方法](mkdocs/customize_postprocess.md)

## 模型列表

<details open markdown>
<summary>文本检测</summary>

- [x] [DBNet](https://github.com/mindspore-lab/mindocr/blob/main/configs/det/dbnet/README_CN.md) (AAAI'2020)
- [x] [DBNet++](https://github.com/mindspore-lab/mindocr/blob/main/configs/det/dbnet/README_CN.md) (TPAMI'2022)
- [x] [PSENet](https://github.com/mindspore-lab/mindocr/blob/main/configs/det/psenet/README_CN.md) (CVPR'2019)
- [x] [EAST](https://github.com/mindspore-lab/mindocr/blob/main/configs/det/east/README_CN.md)(CVPR'2017)
- [x] [FCENet](https://github.com/mindspore-lab/mindocr/blob/main/configs/det/fcenet/README_CN.md) (CVPR'2021)
</details>

<details open markdown>
<summary>文本识别</summary>

- [x] [CRNN](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/crnn/README_CN.md) (TPAMI'2016)
- [x] [CRNN-Seq2Seq/RARE](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/rare/README_CN.md) (CVPR'2016)
- [x] [SVTR](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/svtr/README_CN.md) (IJCAI'2022)
- [x] [MASTER](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/master/README_CN.md) (PR'2019)
- [x] [VISIONLAN](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/visionlan/README_CN.md) (ICCV'2021)
- [ ] [ABINet](https://arxiv.org/abs/2103.06495) (CVPR'2021) [coming soon]
</details>

关于以上模型的具体训练方法和结果，请参见[configs](https://github.com/mindspore-lab/mindocr/blob/main/configs)下各模型子目录的readme文档。

关于[MindSpore Lite](https://www.mindspore.cn/lite)和[ACL](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/aclcppdevg/aclcppdevg_000004.html)模型推理的支持列表，请参见[MindOCR支持模型列表](inference/models_list.md) and [第三方模型推理支持列表](inference/models_list_thirdparty.md)（如PaddleOCR、MMOCR等）。

## 数据集列表

MindOCR提供了[数据格式转换工具](datasets/converters.md) ，以支持不同格式的OCR数据集，支持用户自定义的数据集。
当前已在模型训练评估中验证过的公开OCR数据集如下。

<details open markdown>
<summary>通用OCR数据集</summary>

- [Born-Digital Images](https://rrc.cvc.uab.es/?ch=1) [[download](datasets/borndigital.md)]
- [CASIA-10K](http://www.nlpr.ia.ac.cn/pal/CASIA10K.html) [[download](datasets/casia10k.md)]
- [CCPD](https://github.com/detectRecog/CCPD) [[download](datasets/ccpd.md)]
- [Chinese Text Recognition Benchmark](https://github.com/FudanVI/benchmarking-chinese-text-recognition) [[paper](https://arxiv.org/abs/2112.15093)]  [[download](datasets/chinese_text_recognition.md)]
- [COCO-Text](https://rrc.cvc.uab.es/?ch=5) [[download](datasets/cocotext.md)]
- [CTW](https://ctwdataset.github.io/) [[download](datasets/ctw.md)]
- [ICDAR2015](https://rrc.cvc.uab.es/?ch=4) [[paper](https://rrc.cvc.uab.es/files/short_rrc_2015.pdf)]  [[download](datasets/icdar2015.md)]
- [ICDAR2019 ArT](https://rrc.cvc.uab.es/?ch=14) [[download](datasets/ic19_art.md)]
- [LSVT](https://rrc.cvc.uab.es/?ch=16) [[download](datasets/lsvt.md)]
- [MLT2017](https://rrc.cvc.uab.es/?ch=8) [[paper](https://ieeexplore.ieee.org/abstract/document/8270168)]  [[download](datasets/mlt2017.md)]
- [MSRA-TD500](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500)) [[paper](https://ieeexplore.ieee.org/abstract/document/6247787)]  [[download](datasets/td500.md)]
- [MTWI-2018](https://tianchi.aliyun.com/competition/entrance/231651/introduction) [[download](datasets/mtwi2018.md)]
- [RCTW-17](https://rctw.vlrlab.net/) [[download](datasets/rctw17.md)]
- [ReCTS](https://rrc.cvc.uab.es/?ch=12) [[download](datasets/rects.md)]
- [SCUT-CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector) [[paper](https://www.sciencedirect.com/science/article/pii/S0031320319300664)]  [[download](datasets/ctw1500.md)]
- [SROIE](https://rrc.cvc.uab.es/?ch=13) [[download](datasets/sroie.md)]
- [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset) [[download](datasets/svt.md)]
- [SynText150k](https://github.com/aim-uofa/AdelaiDet) [[paper](https://arxiv.org/abs/2002.10200)]  [[download](datasets/syntext150k.md)]
- [SynthText](https://www.robots.ox.ac.uk/~vgg/data/scenetext/) [[paper](https://www.robots.ox.ac.uk/~vgg/publications/2016/Gupta16/)]  [[download](datasets/synthtext.md)]
- [TextOCR](https://textvqa.org/textocr/) [[download](datasets/textocr.md)]
- [Total-Text](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Dataset) [[paper](https://arxiv.org/abs/1710.10400)]  [[download](datasets/totaltext.md)]
</details>

我们会在更多的数据集上进行模型训练和验证。该列表将持续更新。

## 重要信息

### 更新日志

- 2023/06/07
1. 增加新模型
    - 文本检测[PSENet](https://github.com/mindspore-lab/mindocr/blob/main/configs/det/psenet)
    - 文本检测[EAST](https://github.com/mindspore-lab/mindocr/blob/main/configs/det/east)
    - 文本识别[SVTR](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/svtr)
2. 添加更多基准数据集及其结果
    - [totaltext](datasets/totaltext.md)
    - [mlt2017](datasets/mlt2017.md)
    - [chinese_text_recognition](datasets/chinese_text_recognition.md)
3. 增加断点续训(resume training)功能，可在训练意外中断时使用。如需使用，请在配置文件中`model`字段下增加`resume`参数，允许传入具体路径`resume: /path/to/train_resume.ckpt`或者通过设置`resume: True`来加载在ckpt_save_dir下保存的trian_resume.ckpt
4. 改进检测模块的后处理部分：默认情况下，将检测到的文本多边形重新缩放到原始图像空间，可以通过在`eval.dataset.output_columns`列表中增加"shape_list"实现。
5. 重构在线推理以支持更多模型，详情请参见[README.md](mkdocs/online_inference.md) 。

- 2023/05/15
1. 增加新模型
    - 文本检测[DBNet++](https://github.com/mindspore-lab/mindocr/blob/main/configs/det/dbnet)
    - 文本识别[CRNN-Seq2Seq](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/rare)
    - 在SynthText数据集上预训练的[DBNet](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_synthtext-40655acb.ckpt)
2. 添加更多基准数据集及其结果
    - [SynthText](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c), [MSRA-TD500](datasets/td500.md), [CTW1500](datasets/ctw1500.md)
    - DBNet的更多基准结果可以[在此找到](https://github.com/mindspore-lab/mindocr/blob/main/configs/det/dbnet/README_CN.md).
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
2. 添加梯度累积，支持大批量训练。用法：在yaml配置中添加`gradient_accumulation_steps`，全局批量大小=batch_size * devices * gradient_aaccumulation_steps。示例可参考`configs/rec/crn/crnn_icdar15.yaml`
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

请参考[CONTRIBUTING.md](mkdocs/contributing.md)作为贡献指南，请按照[Model Template and Guideline](mkdocs/customize_model.md)的指引贡献一个适配所有接口的模型，多谢合作。

### 许可

本项目遵从[Apache License 2.0](mkdocs/license.md)开源许可。

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
