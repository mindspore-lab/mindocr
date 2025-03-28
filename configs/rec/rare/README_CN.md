[English](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/rare/README.md) | 中文

# RARE (CRNN-Seq2Seq)
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [Robust Scene Text Recognition with Automatic Rectification](https://arxiv.org/abs/1603.03915)

## 模型描述
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->

识别自然图像中的文本是一个包含许多未解决问题的挑战性任务。与文档中的文字不同，自然图像中的文字通常具有不规则的形状，这是由透视畸变、曲线字符等因素引起的。该论文提出了RARE（Robust Scene Text Recognition with Automatic Rectification），这是一种对不规则文本具有鲁棒性的识别模型。RARE是一种特别设计的深度神经网络，由空间变换网络（STN）和序列识别网络（SRN）组成。在测试中，图像首先通过预测的Thin-Plate-Spline（TPS）变换进行矫正，成为接下来的SRN可以识别的更加“可读”的图像，SRN通过序列识别方法识别文本。研究表明，该模型能够识别多种类型的不规则文本，包括透视文本和曲线文本。RARE是端到端可训练的，只需要图像和相关的文本标签，这使得训练和部署模型在实际系统中变得更加方便。在几个基准数据集上，该模型达到了SOTA性能，充分证明了所提出模型的有效性。 [<a href="#参考文献">1</a>]

<!--- Guideline: If an architecture table/figure is available in the paper, put one here and cite for intuitive illustration. -->

<p align="center">
  <img src="https://user-images.githubusercontent.com/8342575/236731076-f10ae537-c691-4776-8aa3-5a150e14554e.png" width=450 />
</p>
<p align="center">
  <em> 图1. RARE中的SRN结构 [<a href="#参考文献">1</a>] </em>
</p>

## 配套版本

| mindspore  | ascend driver  |   firmware    | cann toolkit/kernel |
|:----------:|:--------------:|:-------------:|:-------------------:|
|   2.5.0    |    24.1.0      |   7.5.0.3.220  |     8.0.0.beta1    |

## 快速开始

### 安装

环境安装教程请参考MindOCR的 [installation instruction](https://github.com/mindspore-lab/mindocr#installation).

### 数据准备

#### 数据集下载

LMDB格式的训练及验证数据集可以从[这里](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0) (出处: [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here))下载。连接中的文件包含多个压缩文件，其中:
- `data_lmdb_release.zip` 包含了**完整**的一套数据集，有训练集(training/），验证集(validation/)以及测试集(evaluation)。
    - `training.zip` 包括两个数据集，分别是 [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/) 和 [SynthText (ST)](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c)
    - `validation.zip` 是多个单独数据集的训练集的一个合集，包括[IC13](http://rrc.cvc.uab.es/?ch=2), [IC15](http://rrc.cvc.uab.es/?ch=4), [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), 和 [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)。
    - `evaluation.zip` 包含多个基准评估数据集，有[IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset), [IC03](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions), [IC13](http://rrc.cvc.uab.es/?ch=2), [IC15](http://rrc.cvc.uab.es/?ch=4), [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf)和 [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html)
- `validation.zip`: 与 data_lmdb_release.zip 中的validation/ 一样。
- `evaluation.zip`: 与 data_lmdb_release.zip 中的evaluation/ 一样。

#### 数据集使用

解压文件后，数据文件夹结构如下：

``` text
data_lmdb_release/
├── evaluation
│   ├── CUTE80
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC03_860
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC03_867
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC13_1015
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── ...
├── training
│   ├── MJ
│   │   ├── MJ_test
│   │   │   ├── data.mdb
│   │   │   └── lock.mdb
│   │   ├── MJ_train
│   │   │   ├── data.mdb
│   │   │   └── lock.mdb
│   │   └── MJ_valid
│   │       ├── data.mdb
│   │       └── lock.mdb
│   └── ST
│       ├── data.mdb
│       └── lock.mdb
└── validation
    ├── data.mdb
    └── lock.mdb
```

在这里，我们使用 `training/` 文件夹下的数据集进行训练，并使用联合数据集 `validation/` 进行验证。训练后，我们使用 `evaluation/` 下的数据集来评估模型的准确性。

**Training:** (total 14,442,049 samples)
- [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/)
  - Train: 21.2 GB, 7224586 samples
  - Valid: 2.36 GB, 802731 samples
  - Test: 2.61 GB, 891924 samples
- [SynthText (ST)](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c)
  - Train: 16.0 GB, 5522808 samples

**Validation:**
- Valid: 138 MB, 6992 samples

**Evaluation:** (total 12,067 samples)
- [CUTE80](http://cs-chan.com/downloads_CUTE80_dataset.html): 8.8 MB, 288 samples
- [IC03_860](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions): 36 MB, 860 samples
- [IC03_867](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions): 4.9 MB, 867 samples
- [IC13_857](http://rrc.cvc.uab.es/?ch=2): 72 MB, 857 samples
- [IC13_1015](http://rrc.cvc.uab.es/?ch=2): 77 MB, 1015 samples
- [IC15_1811](http://rrc.cvc.uab.es/?ch=4): 21 MB, 1811 samples
- [IC15_2077](http://rrc.cvc.uab.es/?ch=4): 25 MB, 2077 samples
- [IIIT5k_3000](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html): 50 MB, 3000 samples
- [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset): 2.4 MB, 647 samples
- [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf): 1.8 MB, 645 samples

### 配置说明

#### 模型训练的数据配置

如欲重现模型的训练，建议修改配置yaml如下：

```yaml
...
train:
  ...
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # 训练数据集根目录
    data_dir: training/                                               # 训练数据集目录，将与`dataset_root`拼接形成完整训练数据集目录
...
eval:
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # 验证数据集根目录
    data_dir: validation/                                             # 验证数据集目录，将与`dataset_root`拼接形成完整验证数据集目录
  ...
```

#### 模型评估的数据配置

我们使用 `evaluation/` 下的数据集作为基准数据集。在**每个单独的数据集**（例如 CUTE80、IC03_860 等）上，我们通过将数据集的目录设置为评估数据集来执行完整评估。这样，我们就得到了每个数据集对应精度的列表，然后报告的精度是这些值的平均值。

如要重现报告的评估结果，您可以：
- 方法 1：对所有单个数据集重复评估步骤：CUTE80、IC03_860、IC03_867、IC13_857、IC131015、IC15_1811、IC15_2077、IIIT5k_3000、SVT、SVTP。然后取平均分。

- 方法 2：将所有基准数据集文件夹放在同一目录下，例如`evaluation/`。并使用脚本`tools/benchmarking/multi_dataset_eval.py`。

1.评估一个特定的数据集

例如，您可以通过修改配置 yaml 来评估数据集“CUTE80”上的模型，如下所示：

```yaml
...
train:
  # 无需修改训练部分的配置，因验证或推理的时候不必使用该部分
...
eval:
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # 训练数据集根目录
    data_dir: evaluation/CUTE80/                                      # 评估数据集目录，将与`dataset_root`拼接形成完整验证或评估数据集目录
  ...
```

通过使用上述配置 yaml 运行 [模型评估](#模型评估) 部分中所述的`tools/eval.py`，您可以获得数据集 CUTE80 的准确度性能。



2. 对同一文件夹下的多个数据集进行评估

假设您已将所有 benckmark 数据集置于 evaluation/ 下，如下所示：

``` text
data_lmdb_release/
├── evaluation
│   ├── CUTE80
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC03_860
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC03_867
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC13_1015
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── ...
```

然后你可以通过如下修改配置yaml来评估每个数据集，并执行脚本`tools/benchmarking/multi_dataset_eval.py`。

```yaml
...
train:
  # NO NEED TO CHANGE ANYTHING IN TRAIN SINCE IT IS NOT USED
...
eval:
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # Root dir of evaluation dataset
    data_dir: evaluation/                                   # Dir of evaluation dataset, concatenated with `dataset_root` to be the complete dir of evaluation dataset
  ...
```

#### 检查配置文件

除了数据集的设置，请同时重点关注以下变量的配置：`system.distribute`, `system.val_while_train`, `common.batch_size`, `train.ckpt_save_dir`, `train.dataset.dataset_root`, `train.dataset.data_dir`, `train.dataset.label_file`,
`eval.ckpt_load_path`, `eval.dataset.dataset_root`, `eval.dataset.data_dir`, `eval.dataset.label_file`, `eval.loader.batch_size`。说明如下：

```yaml
system:
  distribute: True                                                    # 分布式训练为True，单卡训练为False
  amp_level: 'O2'
  seed: 42
  val_while_train: True                                               # 边训练边验证
  drop_overflow_update: True
common:
  ...
  batch_size: &batch_size 512                                         # 训练批大小
...
train:
  ckpt_save_dir: './tmp_rec'                                          # 训练结果（包括checkpoint、每个epoch的性能和曲线图）保存目录
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # 训练数据集根目录
    data_dir: training/                                               # 训练数据集目录，将与`dataset_root`拼接形成完整训练数据集目录
...
eval:
  ckpt_load_path: './tmp_rec/best.ckpt'                               # checkpoint文件路径
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # 验证或评估数据集根目录
    data_dir: validation/                                             # 验证或评估数据集目录，将与`dataset_root`拼接形成完整验证或评估数据集目录
  ...
  loader:
      shuffle: False
      batch_size: 512                                                 # 验证或评估批大小
...
```

**注意:**
- 由于全局批大小 （batch_size x num_devices） 是对结果复现很重要，因此当NPU卡数发生变化时，调整`batch_size`以保持全局批大小不变，或根据新的全局批大小线性调整学习率。


### 模型训练
<!--- Guideline: Avoid using shell script in the command line. Python script preferred. -->

* 分布式训练

使用预定义的训练配置可以轻松重现报告的结果。对于在多个昇腾910设备上的分布式训练，请将配置参数`distribute`修改为True，并运行：

```shell
# 在多个 Ascend 设备上进行分布式训练
# worker_num代表分布式总进程数量。
# local_worker_num代表当前节点进程数量。
# 进程数量即为训练使用的NPU的数量，单机多卡情况下worker_num和local_worker_num需保持一致。
msrun --worker_num=4 --local_worker_num=4 python tools/train.py --config configs/rec/rare/rare_resnet34.yaml

# 经验证，绑核在大部分情况下有性能加速，请配置参数并运行
msrun --bind_core=True --worker_num=4 --local_worker_num=4 python tools/train.py --config configs/rec/rare/rare_resnet34.yaml
```
**注意:** 有关 msrun 配置的更多信息，请参考[此处](https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html).


* 单卡训练

如果要在没有分布式训练的情况下在较小的数据集上训练或微调模型，请将配置参数`distribute`修改为False 并运行：

```shell
# CPU/Ascend 设备上的单卡训练
python tools/train.py --config configs/rec/rare/rare_resnet34.yaml
```

训练结果（包括checkpoint、每个epoch的性能和曲线图）将被保存在yaml配置文件的`ckpt_save_dir`参数配置的目录下，默认为`./tmp_rec`。

### 模型评估

若要评估已训练模型的准确性，可以使用`eval.py`。请在yaml配置文件的`eval`部分将参数`ckpt_load_path`设置为模型checkpoint的文件路径，设置`distribute`为False，然后运行：

```shell
python tools/eval.py --config configs/rec/rare/rare_resnet34.yaml
```
## 评估结果
<!--- Guideline:
Table Format:
- Model: model name in lower case with _ seperator.
- Top-1 and Top-5: Keep 2 digits after the decimal point.
- Params (M): # of model parameters in millions (10^6). Keep 2 digits after the decimal point
- Recipe: Training recipe/configuration linked to a yaml config file. Use absolute url path.
- Download: url of the pretrained model weights. Use absolute url path.
-->

### 精度结果

根据我们的实验，在公开基准数据集（IC03，IC13，IC15，IIIT，SVT，SVTP，CUTE）上的评估结果如下：

<div align="center">

| **model name** | **backbone** | **train dataset** | **params(M)** | **cards** | **batch size** | **jit level** | **graph compile** | **ms/step** | **img/s** | **accuracy** |                                           **recipe**                                            |                                                                                              **weight**                                                                                               |
|:--------------:|:------------:| :---------------: |:-------------:|:---------:|:--------------:| :-----------: |:-----------------:|:-----------:|:---------:|:------------:|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      RARE      | ResNet34_vd  |       MJ+ST       |     25.31     |     4     |      512       |      O2       |     252.62 s      |   180.26    | 11361.43  |    85.24%    | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/rare/rare_resnet34.yaml)  | [ckpt](https://download.mindspore.cn/toolkits/mindocr/rare/rare_resnet34-309dc63e.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/rare/rare_resnet34_ascend-309dc63e-b96c2a4b.mindir)|

</div>

<details open markdown>
  <div align="center">
  <summary>在各个基准数据集上的准确率</summary>

| **model name** |  **backbone**  | **cards** | **IC03_860** | **IC03_867** | **IC13_857** | **IC13_1015** | **IC15_1811** | **IC15_2077** | **IIIT5k_3000** | **SVT** | **SVTP** | **CUTE80** | **average** |
|:--------------:|:--------------:|:---------:|:------------:|:------------:|:------------:|:-------------:|:-------------:|:-------------:|:---------------:|:-------:|:--------:|:----------:|:-----------:|
|      RARE      |  ResNet34_vd   |     1     |    95.12%    |    94.57%    |    94.40%    |    92.81%     |    75.43%     |    69.62%     |     88.17%      | 87.33%  |  78.91%  |   76.04%   |   85.24%    |

   </div>
</details>

**注意:**
- 如需在其他环境配置重现训练结果，请确保全局批量大小与原配置文件保持一致。
- 模型所能识别的字符都是默认的设置，即所有英文小写字母a至z及数字0至9，详细请看[字符词典](#字符词典)
- 模型都是从头开始训练的，无需任何预训练。关于训练和测试数据集的详细介绍，请参考[数据准备](#数据准备)章节。
- RARE的MindIR导出时的输入Shape均为(1, 3, 32, 100)，只能在昇腾卡上使用。

## 字符词典

### 默认设置

在数据处理时，真实文本会根据提供的字符字典转换为标签 ID，字典中键是字符，值是 ID。默认情况下，字典 **"0123456789abcdefghijklmnopqrstuvwxyz"**，这代表着id=0 将对应字符'0'。在默认设置下，字典只考虑数字和小写英文字符，不包括空格。


### 内置词典

Mindocr内置了一部分字典，均放在了 `mindocr/utils/dict/` 位置，可选择合适的字典使用。

- `en_dict.txt` 是一个包含94个字符的英文字典，其中有数字，常用符号以及大小写的英文字母。
- `ch_dict.txt` 是一个包含6623个字符的中文字典，其中有常用的繁简体中文，数字，常用符号以及大小写的英文字母。


### 自定义词典

您也可以自定义一个字典文件 (***.txt)， 放在 `mindocr/utils/dict/` 下，词典文件格式应为每行一个字符的.txt 文件。


如需使用指定的词典，请将参数 `character_dict_path` 设置为字典的路径，并将参数 `num_classes` 改成对应的数量，即字典中字符的数量 + 2。


**注意：**
- 您可以通过将配置文件中的参数 `use_space_char` 设置为 True 来包含空格字符。
- 请记住检查配置文件中的 `dataset->transform_pipeline->RecAttnLabelEncode->lower` 参数的值。如果词典中有大小写字母而且想区分大小写的话，请将其设置为 False。

## 中文识别模型训练

目前，RARE模型支持多语种识别和提供中英预训练模型。详细内容如下

### 中文数据集准备及配置

我们采用公开的中文基准数据集[Benchmarking-Chinese-Text-Recognition](https://github.com/FudanVI/benchmarking-chinese-text-recognition)进行RARE模型的训练和验证。

详细的数据准备和config文件配置方式, 请参考 [中文识别数据集准备](../../../docs/zh/datasets/chinese_text_recognition.md)


### 预训练模型数据集介绍
不同语种的预训练模型采用不同数据集作为预训练，数据来源、训练方式和评估方式可参考 **数据说明**。

| **语种** |                             **数据说明**                             |
| :------: |:----------------------------------------------------------------:|
| 中文 | [中文识别数据集](../../../docs/zh/datasets/chinese_text_recognition.md) |

### 评估结果和预训练权重
模型训练完成后，在测试集不同场景上的准确率评估结果如下。相应的模型配置和预训练权重可通过表中链接下载。

<div align="center">

| **Model** | **Language** | **Backbone** | **Transform Module** | **Scene** | **Web** | **Document** | **Train T.** | **FPS** |                                                     **Recipe**                                                     |                                                                                                **Download**                                                                                                 |
|:---------:|:------------:|:------------:|:--------------------:|:---------:|:-------:|:------------:|:------------:|:-------:|:------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|   RARE    |   Chinese    | ResNet34_vd  |         None         |  62.39%   | 67.02%  |    97.60%    | 414 s/epoch  |  2160   | [rare_resnet34_ch.yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/rare/rare_resnet34_ch.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/rare/rare_resnet34_ch-5f3023e2.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/rare/rare_resnet34_ch_ascend-5f3023e2-11f0d554.mindir) |


</div>

- RARE的MindIR导出时的输入Shape均为(1, 3, 32, 320)，只能在昇腾卡上使用。

### 使用自定义数据集进行训练
您可以在自定义的数据集基于提供的预训练权重进行微调训练, 以在特定场景获得更高的识别准确率，具体步骤请参考文档 [使用自定义数据集训练识别网络](../../../docs/zh/tutorials/training_recognition_custom_dataset.md)。


## MindSpore Lite 推理

请参考[MindOCR 推理](../../../docs/zh/inference/inference_tutorial.md)教程，基于MindSpore Lite在Ascend 310上进行模型的推理，包括以下步骤：

**模型导出**

请先[下载](#2-评估结果)已导出的MindIR文件，或者参考[模型导出](../../../docs/zh/inference/convert_tutorial.md#1-模型导出)教程，使用以下命令将训练完成的ckpt导出为MindIR文件:

```shell
python tools/export.py --model_name_or_config rare_resnet34 --data_shape 32 100 --local_ckpt_path /path/to/local_ckpt.ckpt
# or
python tools/export.py --model_name_or_config configs/rec/rare/rare_resnet34.yaml --data_shape 32 100 --local_ckpt_path /path/to/local_ckpt.ckpt
```

其中，`data_shape`是导出MindIR时的模型输入Shape的height和width，下载链接中MindIR对应的shape值见[注释](#2-评估结果)。

**环境搭建**

请参考[环境安装](../../../docs/zh/inference/environment.md#)教程，配置MindSpore Lite推理运行环境。

**模型转换**

请参考[模型转换](../../../docs/zh/inference/convert_tutorial.md#2-mindspore-lite-mindir-转换)教程，使用`converter_lite`工具对MindIR模型进行离线转换。

**执行推理**

假设在模型转换后得到output.mindir文件，在`deploy/py_infer`目录下使用以下命令进行推理：

```shell
python infer.py \
    --input_images_dir=/your_path_to/test_images \
    --rec_model_path=your_path_to/output.mindir \
    --rec_model_name_or_config=../../configs/rec/rare/rare_resnet34.yaml \
    --res_save_dir=results_dir
```


## 参考文献
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Baoguang Shi, Xinggang Wang, Pengyuan Lyu, Cong Yao, Xiang Bai. Robust Scene Text Recognition with Automatic Rectification. arXiv preprint arXiv:1603.03915, 2016.
