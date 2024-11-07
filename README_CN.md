<!--start-->
<div align="center" markdown>

# MindOCR
</div>

<!--end-->

<div align="center" markdown>

[![CI](https://github.com/mindspore-lab/mindocr/actions/workflows/ci.yml/badge.svg)](https://github.com/mindspore-lab/mindocr/actions/workflows/ci.yml)
[![license](https://img.shields.io/github/license/mindspore-lab/mindocr.svg)](https://github.com/mindspore-lab/mindocr/blob/main/LICENSE)
[![open issues](https://img.shields.io/github/issues/mindspore-lab/mindocr)](https://github.com/mindspore-lab/mindocr/issues)
[![PRs](https://img.shields.io/badge/PRs-welcome-pink.svg)](https://github.com/mindspore-lab/mindocr/pulls)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


[English](README.md) | 中文

[📝简介](#简介) |
[🔨安装教程](#安装教程) |
[🚀快速开始](#快速开始) |
[📚使用教程](#使用教程) |
[🎁模型列表](#模型列表) |
[📰数据集列表](#数据集列表) |
[📖常见问题](#常见问题) |
[🎉更新日志](#更新日志)

</div>

<!--start-->
## 简介
MindOCR是一个基于[MindSpore](https://www.mindspore.cn/en) 框架开发的OCR开源工具箱，集成系列主流文字检测识别的算法、模型，并提供易用的训练和推理工具，可以帮助用户快速开发和应用业界SoTA文本检测、文本识别模型，如DBNet/DBNet++和CRNN/SVTR，满足图像文档理解的需求。


<details open markdown>
<summary> 主要特性 </summary>

- **模块化设计**: MindOCR将OCR任务解耦成多个可配置模块，用户只需修改几行代码，就可以轻松地在定制化的数据和模型上配置训练、评估的全流程；
- **高性能**: MindOCR提供的预训练权重和训练方法可以使其达到OCR任务上具有竞争力的表现；
- **易用性**: MindOCR提供易用工具帮助在真实世界数据中进行文本的检测和识别。
</details>

以下是对应的“mindocr”版本和支持 Mindspore 版本。

| mindocr | mindspore |
|:-------:|:---------:|
| master  |  master   |
|   0.4   |   2.3.0   |
|   0.3   |  2.2.10   |
|   0.1   |    1.8    |


## 安装教程

<details open markdown>

#### MindSpore相关环境准备

MindOCR基于MindSpore AI框架开发，并适配以下框架版本。模型训练场景：

- mindspore [[安装](https://www.mindspore.cn/install)] 请按照mindocr分支安装对应版本MindSpore。
- python >= 3.7
- openmpi 4.0.3 (用于分布式训练与验证)  [[安装](https://www.open-mpi.org/software/ompi/v4.0/)]

Lite推理环境准备请参考[离线推理环境准备](docs/zh/inference/environment.md)。

#### 包依赖

```shell
pip install -r requirements.txt
```

#### 通过源文件安装（推荐）

```shell
git clone https://github.com/mindspore-lab/mindocr.git
cd mindocr
pip install -e .
```
> 使用 `-e` 代表可编辑模式，可以帮助解决潜在的模块导入问题。

#### 通过docker安装

<details>
<summary> 详情 </summary>

目前提供的docker，环境信息如下
 - 操作系统版本：Euler2.8
 - CANN版本：7.0
 - Python版本：3.9
 - MindSpore 版本：2.2.10
 - MindSpore Lite 版本：2.2.10

使用docker安装，根据以下步骤：

1. 下载docker
    - 910：
        ```bash
        docker pull swr.cn-central-221.ovaijisuan.com/mindocr/mindocr_dev_910_ms_2_2_10_cann7_0_py39:v1
        ```
    - 910*:
        ```bash
        docker pull swr.cn-central-221.ovaijisuan.com/mindocr/mindocr_dev_ms_2_2_10_cann7_0_py39:v1
        ```
2. 新建容器
    ```bash
    docker_name="temp_mindocr"
    # 910
    image_name="swr.cn-central-221.ovaijisuan.com/mindocr/mindocr_dev_910_ms_2_2_10_cann7_0_py39:v1"
    # 910*
    image_name="swr.cn-central-221.ovaijisuan.com/mindocr/mindocr_dev_ms_2_2_10_cann7_0_py39:v1"

    docker run --privileged --name ${docker_name} \
        --tmpfs /tmp \
        --tmpfs /run \
        -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
        --device=/dev/davinci1 \
        --device=/dev/davinci2 \
        --device=/dev/davinci3 \
        --device=/dev/davinci4 \
        --device=/dev/davinci5 \
        --device=/dev/davinci6 \
        --device=/dev/davinci7 \
        --device=/dev/davinci_manager \
        --device=/dev/hisi_hdc \
        --device=/dev/devmm_svm \
        -v /etc/localtime:/etc/localtime \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        --shm-size 800g \
        --cpus 96 \
        --security-opt seccomp=unconfined \
        --network=bridge -itd ${image_name} bash
    ```

3. 进入容器
    ```bash
    # 设置docker id
    container_id="your docker id"
    docker exec -it --user root $container_id bash
    ```

4. 设置环境变量
    进入容器后，设置环境变量：
    ```bash
    source env_setup.sh
    ```

</details>

#### 通过PyPI安装
```shell
pip install mindocr
```

>由于此项目正在积极开发中，从PyPI安装的版本目前已过期，我们将很快更新，敬请期待。

</details>

## 快速开始

### 1. 文字检测和识别示例

安装完MindOCR后，我们就很方便地进行任意图像的文本检测和识别，如下。

```shell
python tools/infer/text/predict_system.py --image_dir {path_to_img or dir_to_imgs} \
                                          --det_algorithm DB++  \
                                          --rec_algorithm CRNN  \
                                          --visualize_output True
```

运行结束后，结果将被默认保存在`./inference_results`路径，可视化结果如下：
<p align="center">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/c1f53970-8618-4039-994f-9f6dc1eee1dd" width=600 />
</p>
<p align="center">
  <em> 文本检测、识别结果可视化 </em>
</p>

可以看到图像中的文字块均被检测出来并正确识别。更详细的用法介绍，请参考推理[教程](#使用教程)。

### 2. 模型训练、评估与推理-快速指南

使用`tools/train.py`脚本可以进行OCR模型训练，该脚本可支持文本检测和识别模型训练。
```shell
python tools/train.py --config {path/to/model_config.yaml}
```
`--config` 参数用于指定yaml文件的路径，该文件定义要训练的模型和训练策略，包括数据处理流程、优化器、学习率调度器等。

MindOCR在`configs`文件夹中提供系列SoTA的OCR模型及其训练策略，用户可以快速将其适配到自己的任务或数据集上，参考例子如下

```shell
# train text detection model DBNet++ on icdar15 dataset
python tools/train.py --config configs/det/dbnet/dbpp_r50_icdar15.yaml
```
```shell
# train text recognition model CRNN on icdar15 dataset
python tools/train.py --config configs/rec/crnn/crnn_icdar15.yaml
```

使用`tools/eval.py` 脚本可以评估已训练好的模型，如下所示：
```shell
python tools/eval.py \
    --config {path/to/model_config.yaml} \
    --opt eval.dataset_root={path/to/your_dataset} eval.ckpt_load_path={path/to/ckpt_file}
```

使用`tools/infer/text/predict_system.py` 脚本可进行模型的在线推理，如下所示：
```shell
python tools/infer/text/predict_system.py --image_dir {path_to_img or dir_to_imgs} \
                                          --det_algorithm DB++  \
                                          --rec_algorithm CRNN  \
                                          --visualize_output True
```

更多使用方法，请参考[使用教程](#使用教程)中的模型训练、推理章节。

### 3. 模型离线推理

你可以在MindOCR中对**MindOCR原生模型**或**第三方模型**（如PaddleOCR、MMOCR等）进行MindSpore Lite推理。详情请参考[模型离线推理教程](docs/zh/inference/inference_tutorial.md)。

## 使用教程

- 数据集
    - [数据集准备](docs/zh/datasets/converters.md)
    - [数据增强策略](docs/zh/tutorials/transform_tutorial.md)
- 模型训练
    - [Yaml配置文件](docs/zh/tutorials/yaml_configuration.md)
    - [文本检测](docs/zh/tutorials/training_detection_custom_dataset.md)
    - [文本识别](docs/zh/tutorials/training_recognition_custom_dataset.md)
    - [分布式训练](docs/zh/tutorials/distribute_train.md)
    - [进阶技巧：梯度累积，EMA，断点续训等](docs/zh/tutorials/advanced_train.md)
- 使用MindSpore进行在线推理
    - [基于Python的OCR在线推理](tools/infer/text/README_CN.md)
- 使用MindSpore Lite进行离线推理
    - [模型离线推理教程](docs/zh/inference/inference_tutorial.md)
- 开发者指南
    - [如何自定义数据集](mindocr/data/README_CN.md)
    - [如何自定义数据增强方法](mindocr/data/transforms/README_CN.md)
    - [如何创建新的OCR模型](mindocr/models/README_CN.md)
    - [如何自定义后处理方法](mindocr/postprocess/README_CN.md)

## 模型列表

<details open markdown>
<summary>文本检测</summary>

- [x] [DBNet](configs/det/dbnet/README_CN.md) (AAAI'2020)
- [x] [DBNet++](configs/det/dbnet/README_CN.md) (TPAMI'2022)
- [x] [PSENet](configs/det/psenet/README_CN.md) (CVPR'2019)
- [x] [EAST](configs/det/east/README_CN.md)(CVPR'2017)
- [x] [FCENet](configs/det/fcenet/README_CN.md) (CVPR'2021)
</details>

<details open markdown>
<summary>文本识别</summary>

- [x] [CRNN](configs/rec/crnn/README_CN.md) (TPAMI'2016)
- [x] [CRNN-Seq2Seq/RARE](configs/rec/rare/README_CN.md) (CVPR'2016)
- [x] [SVTR](configs/rec/svtr/README_CN.md) (IJCAI'2022)
- [x] [MASTER](configs/rec/master/README_CN.md) (PR'2019)
- [x] [VISIONLAN](configs/rec/visionlan/README_CN.md) (ICCV'2021)
- [x] [RobustScanner](configs/rec/robustscanner/README_CN.md) (ECCV'2020)
- [x] [ABINet](configs/rec/abinet/README_CN.md) (CVPR'2021)
</details>

<details open markdown>
<summary>版面分析</summary>

- [x] [YOLOv8](configs/layout/yolov8/README_CN.md) ([Ultralytics Inc.](https://github.com/ultralytics/ultralytics))
</details>

<details open markdown>
<summary>关键信息抽取</summary>

- [x] [LayoutXLM](configs/kie/vi_layoutxlm/README_CN.md) (arXiv'2021)
- [x] [LayoutLMv3](configs/kie/layoutlmv3/README_CN.md) (arXiv'2022)

</details>

<details open markdown>
<summary>表格识别</summary>

- [x] [TableMaster](configs/table/README_CN.md) (arXiv'2021)

</details>

<details open markdown>
<summary>OCR大模型</summary>

- [x] [Vary](configs/llm/vary/README_CN.md) (arXiv'2023)

</details>


关于以上模型的具体训练方法和结果，请参见[configs](https://github.com/mindspore-lab/mindocr/blob/main/configs)下各模型子目录的readme文档。

[MindSpore Lite](https://www.mindspore.cn/lite)模型推理的支持列表，
请参见[MindOCR原生模型推理支持列表](docs/zh/inference/mindocr_models_list.md) 和 [第三方模型推理支持列表](docs/zh/inference/thirdparty_models_list.md)（如PaddleOCR）。

## 数据集列表

MindOCR提供了[数据格式转换工具](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters) ，以支持不同格式的OCR数据集，支持用户自定义的数据集。
当前已在模型训练评估中验证过的公开OCR数据集如下。

<details close markdown>
<summary>通用OCR数据集</summary>

- [Born-Digital Images](https://rrc.cvc.uab.es/?ch=1) [[download](docs/zh/datasets/borndigital.md)]
- [CASIA-10K](http://www.nlpr.ia.ac.cn/pal/CASIA10K.html) [[download](docs/zh/datasets/casia10k.md)]
- [CCPD](https://github.com/detectRecog/CCPD) [[download](docs/zh/datasets/ccpd.md)]
- [Chinese Text Recognition Benchmark](https://github.com/FudanVI/benchmarking-chinese-text-recognition) [[paper](https://arxiv.org/abs/2112.15093)] [[download](docs/zh/datasets/chinese_text_recognition.md)]
- [COCO-Text](https://rrc.cvc.uab.es/?ch=5) [[download](docs/zh/datasets/cocotext.md)]
- [CTW](https://ctwdataset.github.io/) [[download](docs/zh/datasets/ctw.md)]
- [ICDAR2015](https://rrc.cvc.uab.es/?ch=4) [[paper](https://rrc.cvc.uab.es/files/short_rrc_2015.pdf)] [[download](docs/zh/datasets/icdar2015.md)]
- [ICDAR2019 ArT](https://rrc.cvc.uab.es/?ch=14) [[download](docs/zh/datasets/ic19_art.md)]
- [LSVT](https://rrc.cvc.uab.es/?ch=16) [[download](docs/zh/datasets/lsvt.md)]
- [MLT2017](https://rrc.cvc.uab.es/?ch=8) [[paper](https://ieeexplore.ieee.org/abstract/document/8270168)] [[download](docs/zh/datasets/mlt2017.md)]
- [MSRA-TD500](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500)) [[paper](https://ieeexplore.ieee.org/abstract/document/6247787)] [[download](docs/zh/datasets/td500.md)]
- [MTWI-2018](https://tianchi.aliyun.com/competition/entrance/231651/introduction) [[download](docs/zh/datasets/mtwi2018.md)]
- [RCTW-17](https://rctw.vlrlab.net/) [[download](docs/zh/datasets/rctw17.md)]
- [ReCTS](https://rrc.cvc.uab.es/?ch=12) [[download](docs/zh/datasets/rects.md)]
- [SCUT-CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector) [[paper](https://www.sciencedirect.com/science/article/pii/S0031320319300664)] [[download](docs/zh/datasets/ctw1500.md)]
- [SROIE](https://rrc.cvc.uab.es/?ch=13) [[download](docs/zh/datasets/sroie.md)]
- [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset) [[download](docs/zh/datasets/svt.md)]
- [SynText150k](https://github.com/aim-uofa/AdelaiDet) [[paper](https://arxiv.org/abs/2002.10200)] [[download](docs/zh/datasets/syntext150k.md)]
- [SynthText](https://www.robots.ox.ac.uk/~vgg/data/scenetext/) [[paper](https://www.robots.ox.ac.uk/~vgg/publications/2016/Gupta16/)] [[download](docs/zh/datasets/synthtext.md)]
- [TextOCR](https://textvqa.org/textocr/) [[download](docs/zh/datasets/textocr.md)]
- [Total-Text](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Dataset) [[paper](https://arxiv.org/abs/1710.10400)] [[download](docs/zh/datasets/totaltext.md)]

</details>

<details close markdown>
<summary>版面分析数据集</summary>

- [PublayNet](https://github.com/ibm-aur-nlp/PubLayNet) [[paper](https://arxiv.org/abs/1908.07836)] [[download](https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/publaynet.tar.gz)]

</details>

<details close markdown>
<summary>关键信息抽取数据集</summary>

- [XFUND](https://github.com/doc-analysis/XFUND) [[paper](https://aclanthology.org/2022.findings-acl.253/)] [[download](https://github.com/doc-analysis/XFUND/releases/tag/v1.0)]

</details>

<details close markdown>
<summary>表格识别数据集</summary>

- [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet) [[paper](https://arxiv.org/pdf/1911.10683.pdf)] [[download](https://dax-cdn.cdn.appdomain.cloud/dax-pubtabnet/2.0.0/pubtabnet.tar.gz)]

</details>

我们会在更多的数据集上进行模型训练和验证。该列表将持续更新。

## 常见问题
关于配置环境、使用mindocr遇到的高频问题，可以参考[常见问题](docs/zh/tutorials/frequently_asked_questions.md)。

## 重要信息

### 更新日志
<details close markdown>
<summary>详细</summary>

- 2023/04/01
1. 增加新模型
    - 关键信息抽取[LayoutLMv3](configs/kie/layoutlmv3/)

- 2024/03/20
1. 增加新模型
    - OCR大模型[Vary-toy](configs/llm/vary/vary_toy.yaml)，支持基于通义千问1.8B LLM的检测和OCR功能

- 2023/12/25
1. 增加新模型
    - 表格识别[TableMaster](configs/table/table_master.yaml)
2. 添加更多基准数据集及其结果
    - [PubTabNet](configs/table/README_CN.md)

- 2023/12/14
1. 增加新模型
    - 关键信息抽取[LayoutXLM](configs/kie/layoutxlm)
    - 关键信息抽取[VI-LayoutXLM](configs/kie/vi_layoutxlm)
    - 文本检测[PP-OCRv3 DBNet](configs/det/dbnet/db_mobilenetv3_ppocrv3.yaml)和文本识别[PP-OCRv3 SVTR](configs/rec/svtr/svtr_ppocrv3_ch.yaml)，支持在线推理和微调训练
2. 添加更多基准数据集及其结果
    - [XFUND](configs/kie/vi_layoutxlm/README_CN.md)
3. 昇腾910硬件多规格支持：DBNet ResNet-50、DBNet++ ResNet-50、CRNN VGG7、SVTR-Tiny、FCENet、ABINet
- 2023/11/28
1. 增加支持PP-OCRv4模型离线推理
    - 文本检测 [PP-OCRv4 DBNet](deploy/py_infer/src/configs/det/ppocr/ch_PP-OCRv4_det_cml.yaml)和文本识别 [PP-OCRv4 CRNN](deploy/py_infer/src/configs/rec/ppocr/ch_PP-OCRv4_rec_distillation.yaml)，支持离线推理
2. 修复第三方模型离线推理bug
- 2023/11/17
1. 增加新模型
    - 版面分析[YOLOv8](configs/layout/yolov8)
2. 添加更多基准数据集及其结果
    - [PublayNet](configs/layout/yolov8/README_CN.md)
- 2023/07/06
1. 增加新模型
    - 文本识别 [RobustScanner](configs/rec/robustscanner)
- 2023/07/05
1. 增加新模型
    - 文本识别 [VISIONLAN](configs/rec/visionlan)
- 2023/06/29
1. 新增2个SoTA模型
    - 文本检测 [FCENet](configs/det/fcenet)
    - 文本识别 [MASTER](configs/rec/master)
- 2023/06/07
1. 增加新模型
    - 文本检测 [PSENet](configs/det/psenet)
    - 文本检测 [EAST](configs/det/east)
    - 文本识别 [SVTR](configs/rec/svtr)
2. 添加更多基准数据集及其结果
    - [totaltext](docs/zh/datasets/totaltext.md)
    - [mlt2017](docs/zh/datasets/mlt2017.md)
    - [chinese_text_recognition](docs/zh/datasets/chinese_text_recognition.md)
3. 增加断点续训(resume training)功能，可在训练意外中断时使用。如需使用，请在配置文件中`model`字段下增加`resume`参数，允许传入具体路径`resume: /path/to/train_resume.ckpt`或者通过设置`resume: True`来加载在ckpt_save_dir下保存的trian_resume.ckpt
4. 改进检测模块的后处理部分：默认情况下，将检测到的文本多边形重新缩放到原始图像空间，可以通过在`eval.dataset.output_columns`列表中增加"shape_list"实现。
5. 重构在线推理以支持更多模型，详情请参见[README.md](tools/infer/text/README_CN.md) 。

- 2023/05/15
1. 增加新模型
    - 文本检测 [DBNet++](configs/det/dbnet)
    - 文本识别 [CRNN-Seq2Seq](configs/rec/rare)
    - 在SynthText数据集上预训练的[DBNet](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_synthtext-40655acb.ckpt)
2. 添加更多基准数据集及其结果
    - [SynthText](docs/zh/datasets/synthtext.md), [MSRA-TD500](docs/zh/datasets/td500.md), [CTW1500](docs/zh/datasets/ctw1500.md)
    - DBNet的更多基准结果可以[在此找到](configs/det/dbnet/README_CN.md).
3. 添加用于保存前k个checkpoint的checkpoint manager并改进日志。
4. Python推理代码重构。
5. Bug修复：对大型数据集使用平均损失meter，在AMP训练中对ctcloss禁用`pred_cast_fp32`，修复存在无效多边形的错误。

- 2023/05/04
1. 支持加载自定义的预训练checkpoint， 通过在yaml配置中将`model-pretrained`设置为checkpoint url或本地路径来使用。
2. 支持设置执行包括旋转和翻转在内的数据增强操作的概率。
3. 为模型训练添加EMA功能，可以通过在yaml配置中设置`train-ema`（默认值：False）和`train-ema_decay`来启用。
4. 参数修改：`num_columns_to_net` -> `net_input_column_index`: 输入网络的columns数量改为输入网络的columns索引
5. 参数修改：`num_columns_of_labels` -> `label_column_index`: 用索引替换数量，以表示label的位置。

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
</details>

### 如何贡献

我们欢迎包括问题单和PR在内的所有贡献，来让MindOCR变得更好。

请参考[CONTRIBUTING.md](CONTRIBUTING_CN.md)作为贡献指南，请按照[Model Template and Guideline](mindocr/models/README_CN.md)的指引贡献一个适配所有接口的模型，多谢合作。

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
<!--end-->
