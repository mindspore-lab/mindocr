[English](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/robustscanner/README.md) | 中文

# RobustScanner
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [RobustScanner: Dynamically Enhancing Positional Clues for Robust Text Recognition](https://arxiv.org/pdf/2007.07542.pdf)

## 1. 模型描述
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->

RobustScanner是具有注意力机制的编码器-解码器文字识别算法，本作作者通过对当时主流方法编解码器识别框架的研究，发现文字在解码过程中，不仅依赖上下文信息，还会利用位置信息。而大多数方法在解码过程中都过度依赖语义信息，导致存在较为严重的注意力偏移问题，对于没有语义信息或者弱语义信息的文本识别效果不佳。

针对于此，作者提出了一个新颖的位置增强分支，并动态地将其输出与解码器注意力模块的输出融合。位置增强分支包含了位置感知模块、位置编码层和注意力模块。其中位置感知模块的作用是对编码器的输出特征图进行增强，使得其可以编码丰富的位置信息。位置编码层接受当前解码步数作为输入，将其编码为一个向量。

总体来看，RobustScanner模型由编码器和解码器两部分组成，编码器使用了ResNet-31来提取图像的特征。解码器包含混合分支和位置增强分支，两个分支的输出经过动态融合模块合并，输出预测结果。基于对位置信息的特殊设计，RobustScanner在规则和不规则文本识别基准测试上取得了当时最先进的结果，在无上下文基准测试上没有太大的性能下降，从而验证了其在上下文和无上下文应用程序场景中的鲁棒性。

<p align="center">
  <img src="https://github.com/tonytonglt/mindocr/assets/54050944/7c11121b-1962-4d29-93b6-5c9533992b8f" width=640 />
</p>
<p align="center">
  <em> 图1. RobustScanner整体架构图 [<a href="#参考文献">1</a>] </em>
</p>


## 2. 评估结果
<!--- Guideline:
Table Format:
- Model: model name in lower case with _ seperator.
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Top-1 and Top-5: Keep 2 digits after the decimal point.
- Params (M): # of model parameters in millions (10^6). Keep 2 digits after the decimal point
- Recipe: Training recipe/configuration linked to a yaml config file. Use absolute url path.
- Download: url of the pretrained model weights. Use absolute url path.
-->

### 训练端
根据我们的实验，在公开基准数据集（IC03，IC13，IC15，IIIT，SVT，SVTP，CUTE）上的评估结果如下：


<div align="center">

|    **模型**     |    **环境配置**    | **骨干网络**  | **平均准确率** |       **训练时间**        | **FPS** | **ms/step** |                                                     **配置文件**                                                     |                                                                                                             **模型权重下载**                                                                                                              |
|:-------------:|:--------------:|:---------:|:---------:|:---------------------:|:-------:|:-----------:|:----------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| RobustScanner | D910x4-MS2.0-G | ResNet-31 |  87.86%   |     22560 s/epoch     |   310   |     825     | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/robustscanner/robustscanner_resnet31.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/robustscanner/robustscanner_resnet31-f27eab37.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/robustscanner/robustscanner_resnet31-f27eab37-158bde10.mindir) |
</div>

注：除了使用MJSynth（部分）和SynthText（部分）两个文字识别数据集外，还加入了SynthAdd数据，和部分真实数据，具体数据细节可以参考论文或[这里](#312-数据集下载)。

<details open markdown>
  <div align="center">
  <summary>在各个基准数据集上的准确率</summary>

  | **模型** | **骨干网络** | **IIIT5k** | **SVT** | **IC13** | **IC15** | **SVTP** | **CUTE80** | **平均准确率** |
  | :------: | :------: |:----------:|:-------:|:--------:|:--------:|:--------:|:----------:|:---------:|
  | RobustScanner  | ResNet-31 |   95.50%   | 92.12%  |  94.29%  |  73.33%  |  82.33%  |   89.58%   |  87.86%   |
  </div>
</details>

**注意:**
- 环境配置：训练的环境配置表示为 {处理器}x{处理器数量}-{MS模式}，其中 Mindspore 模式可以是 G-graph 模式或 F-pynative 模式。例如，D910x4-MS2.0-G 用于使用图形模式在4张昇腾910 NPU上依赖Mindspore2.0版本进行训练。
- 如需在其他环境配置重现训练结果，请确保全局批量大小与原配置文件保持一致。
- 模型使用90个字符的英文字典en_dict90.txt，其中有数字，常用符号以及大小写的英文字母，详细请看[4. 字符词典](#4-字符词典)
- 模型都是从头开始训练的，无需任何预训练。关于训练和测试数据集的详细介绍，请参考[数据集下载及使用](#312-数据集下载)章节。
- RobustScanner的MindIR导出时的输入Shape均为(1, 3, 48, 160)。

## 3. 快速开始
### 3.1 环境及数据准备

#### 3.1.1 安装
环境安装教程请参考MindOCR的 [installation instruction](https://github.com/mindspore-lab/mindocr#installation).

#### 3.1.2 数据集下载
本RobustScanner训练、验证使用的数据集参考了mmocr和PaddleOCR所使用的数据集对文献算法进行复现，在此非常感谢mmocr和PaddleOCR，提高了本repo的复现效率。

数据集细节如下：
<div align="center">

|  **训练集**  |    **样本数量**    | **重复次数** | **数据类型** |
|:---------:|:--------------:|:--------:|:--------:|
| icdar2013 |      848       |    20    |    真实    |
| icdar2015 |      4468      |    20    |    真实    |
| coco_text |     42142      |    20    |    真实    |
|  IIIT5K   |      2000      |    20    |    真实    |
| SynthText |    2400000     |    1     |    合成    |
| SynthAdd  |    1216889     |    1     |    合成    |
|  Syn90k   |    2400000     |    1     |    合成    |

</div>
注：SynthText和Syn90k均是随机挑选240万个样本。

上表中LMDB格式的训练及验证数据集可以从这里下载: [训练集](https://aistudio.baidu.com/aistudio/datasetdetail/138433)、[验证集](https://aistudio.baidu.com/aistudio/datasetdetail/138872)

连接中的文件包含多个压缩文件，其中:
- 训练集
  - `training_lmdb_real.zip`: 包含了IIIT5K, icdar2013, icdar2015, coco_text四个真实数据集，这些数据会在训练时重复20次；
  - `SynthAdd.zip`: 包含SynthAdd数据集的完整数据；
  - `synth90K_shuffle.zip`: 包含Synth90k数据集中随机挑选的240万个样本；
  - `SynthText800K_shuffle_xxx_xxx.zip`: 1_200共5个zip文件，包含SynthText数据集中随机挑选的240万个样本。
- 验证集
  - `testing_lmdb.zip`: 包含了评估模型使用的CUTE80, icdar2013, icdar2015, IIIT5k, SVT, SVTP六个数据集。
#### 3.1.3 数据集使用

数据文件夹按照如下结构进行解压：

``` text
data/
├── training
│   ├── real_data
│   │   ├── repeat1
│   │   │   ├── COCO_Text
│   │   │   │   ├── data.mdb
│   │   │   │   └── lock.mdb
│   │   │   ├── ICDAR2013
│   │   │   │   ├── data.mdb
│   │   │   │   └── lock.mdb
│   │   │   ├── ICDAR2015
│   │   │   │   ├── data.mdb
│   │   │   │   └── lock.mdb
│   │   │   └── IIIT5K
│   │   │       ├── data.mdb
│   │   │       └── lock.mdb
│   │   ├── repeat2
│   │   │   ├── COCO_Text
│   │   │   ├── ICDAR2013
│   │   │   ├── ICDAR2015
│   │   │   └── IIIT5K
│   │   │
│   │   ├── ...
│   │   │
│   │   └── repeat20
│   │       ├── COCO_Text
│   │       ├── ICDAR2013
│   │       ├── ICDAR2015
│   │       └── IIIT5K
│   └── synth_data
│       ├── synth90K_shuffle
│       │   ├── data.mdb
│       │   └── lock.mdb
│       ├── SynthAdd
│       │   ├── data.mdb
│       │   └── lock.mdb
│       ├── SynthText800K_shuffle_1_40
│       │   ├── data.mdb
│       │   └── lock.mdb
│       ├── SynthText800K_shuffle_41_80
│       │   ├── data.mdb
│       │   └── lock.mdb
│       └── ...
└── evaluation
    ├── CUTE80
    │   ├── data.mdb
    │   └── lock.mdb
    ├── IC13_1015
    │   ├── data.mdb
    │   └── lock.mdb
    ├── IC15_2077
    │   ├── data.mdb
    │   └── lock.mdb
    ├── IIIT5k_3000
    │   ├── data.mdb
    │   └── lock.mdb
    ├── SVT
    │   ├── data.mdb
    │   └── lock.mdb
    └── SVTP
        ├── data.mdb
        └── lock.mdb

```
在这里，我们使用 `training/` 文件夹下的数据集进行训练，并使用 `evaluation/` 下的数据集来进行模型的验证和评估。为方便存储和使用，所有数据均为lmdb格式

**模型训练的数据配置**

如欲重现模型的训练，建议修改配置yaml如下：

```yaml
...
train:
  ...
  dataset:
    type: LMDBDataset
    dataset_root: path/to/data/                           # 数据集根目录
    data_dir: training/                                   # 训练数据集目录，将与`dataset_root`拼接形成完整训练数据集目录
...
eval:
  dataset:
    type: LMDBDataset
    dataset_root: path/to/data/                           # 数据集根目录
    data_dir: evaluation/                                 # 验证数据集目录，将与`dataset_root`拼接形成完整验证数据集目录
  ...
```

**模型评估的数据配置**

我们使用 `evaluation/` 下的数据集作为基准数据集。在**每个单独的数据集**（例如 CUTE80、IC13_1015 等）上，我们通过将数据集的目录设置为评估数据集来执行完整评估。这样，我们就得到了每个数据集对应精度的列表，然后报告的精度是这些值的平均值。

如要重现报告的评估结果，您可以：
- 方法 1：对所有单个数据集重复评估步骤：CUTE80、IC13_1015、IC15_2077、IIIT5k_3000、SVT、SVTP。然后取平均分。

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
    dataset_root: path/to/data/                           # 数据集根目录
    data_dir: evaluation/CUTE80/                          # 评估数据集目录，将与`dataset_root`拼接形成完整验证或评估数据集目录
  ...
```

通过使用上述配置 yaml 运行 [模型评估](#33-模型评估) 部分中所述的`tools/eval.py`，您可以获得数据集 CUTE80 的准确度性能。


2. 对同一文件夹下的多个数据集进行评估

假设您已将所有 benckmark 数据集置于 evaluation/ 下，如下所示：

``` text
data/
├── evaluation
│   ├── CUTE80
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC13_1015
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC15_2077
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IIIT5k_3000
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── ...
```

然后你可以通过如下修改配置yaml来评估每个数据集，并执行脚本`tools/benchmarking/multi_dataset_eval.py`。

```yaml
...
train:
  # 无需修改训练部分的配置，因验证或推理的时候不必使用该部分
...
eval:
  dataset:
    type: LMDBDataset
    dataset_root: path/to/data/                           # 数据集根目录
    data_dir: evaluation/                                 # 评估数据集目录，将与`dataset_root`拼接形成完整验证或评估数据集目录
  ...
```

#### 3.1.4 检查配置文件
除了数据集的设置，请同时重点关注以下变量的配置：`system.distribute`, `system.val_while_train`, `common.batch_size`, `train.ckpt_save_dir`, `train.dataset.dataset_root`, `train.dataset.data_dir`,
`eval.ckpt_load_path`, `eval.dataset.dataset_root`, `eval.dataset.data_dir`, `eval.loader.batch_size`。说明如下：

```yaml
system:
  distribute: True                                                    # 分布式训练为True，单卡训练为False
  amp_level: 'O0'
  seed: 42
  val_while_train: True                                               # 边训练边验证
  drop_overflow_update: False
common:
  ...
  batch_size: &batch_size 64                                          # 训练批大小
  ...
train:
  ckpt_save_dir: './tmp_rec'                                          # 训练结果（包括checkpoint、每个epoch的性能和曲线图）保存目录
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: path/to/data/                                       # 训练数据集根目录
    data_dir: training/                                               # 训练数据集目录，将与`dataset_root`拼接形成完整训练数据集目录
...
eval:
  ckpt_load_path: './tmp_rec/best.ckpt'                               # checkpoint文件路径
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: path/to/data/                                       # 验证或评估数据集根目录
    data_dir: evaluation/                                             # 验证或评估数据集目录，将与`dataset_root`拼接形成完整验证或评估数据集目录
  ...
  loader:
      shuffle: False
      batch_size: 64                                                  # 验证或评估批大小
...
```

**注意:**
- 由于全局批大小 （batch_size * num_devices） 是对结果复现很重要，因此当GPU/NPU卡数发生变化时，调整`batch_size`以保持全局批大小不变，或根据新的全局批大小线性调整学习率。

**使用自定义数据集进行训练**
- 您可以在自定义的数据集基于提供的预训练权重进行微调训练, 以在特定场景获得更高的识别准确率，具体步骤请参考文档 [使用自定义数据集训练识别网络](../../../docs/cn/tutorials/training_recognition_custom_dataset.md)。


### 3.2 模型训练
<!--- Guideline: Avoid using shell script in the command line. Python script preferred. -->

* 分布式训练

使用预定义的训练配置可以轻松重现报告的结果。对于在多个昇腾910设备上的分布式训练，请将配置参数`distribute`修改为True，并运行：

```shell
# 在多个 GPU/Ascend 设备上进行分布式训练
mpirun --allow-run-as-root -n 4 python tools/train.py --config configs/rec/robustscanner/robustscanner_resnet31.yaml
```


* 单卡训练

如果要在没有分布式训练的情况下在较小的数据集上训练或微调模型，请将配置参数`distribute`修改为False 并运行：

```shell
# CPU/GPU/Ascend 设备上的单卡训练
python tools/train.py --config configs/rec/robustscanner/robustscanner_resnet31.yaml
```

训练结果（包括checkpoint、每个epoch的性能和曲线图）将被保存在yaml配置文件的`ckpt_save_dir`参数配置的目录下，默认为`./tmp_rec`。

### 3.3 模型评估

若要评估已训练模型的准确性，可以使用`eval.py`。请在yaml配置文件的`eval`部分将参数`ckpt_load_path`设置为模型checkpoint的文件路径，设置`distribute`为False，然后运行：

```shell
python tools/eval.py --config configs/rec/robustscanner/robustscanner_resnet31.yaml
```

## 4. 字符词典

### 默认设置

在数据处理时，真实文本会根据提供的字符字典转换为标签 ID，字典中键是字符，值是 ID。默认情况下，字典 **"0123456789abcdefghijklmnopqrstuvwxyz"**，这代表着id=0 将对应字符'0'。在默认设置下，字典只考虑数字和小写英文字符，不包括空格。


### 内置词典

Mindocr内置了一部分字典，均放在了 `mindocr/utils/dict/` 位置，可选择合适的字典使用。

- `en_dict90.txt` 是一个包含90个字符的英文字典，其中有数字，常用符号以及大小写的英文字母。
- `en_dict.txt` 是一个包含94个字符的英文字典，其中有数字，常用符号以及大小写的英文字母。
- `ch_dict.txt` 是一个包含6623个字符的中文字典，其中有常用的繁简体中文，数字，常用符号以及大小写的英文字母。


### 自定义词典

您也可以自定义一个字典文件 (***.txt)， 放在 `mindocr/utils/dict/` 下，词典文件格式应为每行一个字符的.txt 文件。


如需使用指定的词典，请将参数 `character_dict_path` 设置为字典的路径，并修改如下参数`model->head->out_channels`改为字典中字符的数量+3，`model->head->start_idx`改为字典中字符的数量+1，`model->head->padding_idx`改为字典中字符的数量+2，`loss->ignore_index`改为字典中字符的数量+2。

```yaml
...
model:
  type: rec
  transform: null
  backbone:
    name: rec_resnet31
    pretrained: False
  head:
    name: RobustScannerHead
    out_channels: 93                            # 修改为字典中字符的数量+3
    enc_outchannles: 128
    hybrid_dec_rnn_layers: 2
    hybrid_dec_dropout: 0.
    position_dec_rnn_layers: 2
    start_idx: 91                               # 修改为字典中字符的数量+1
    mask: True
    padding_idx: 92                             # 修改为字典中字符的数量+2
    encode_value: False
    max_text_len: *max_text_len
...

loss:
  name: SARLoss
  ignore_index: 92                              # 修改为字典中字符的数量+2

...
```

**注意：**
- 您可以通过将配置文件中的参数 `use_space_char` 设置为 True 来包含空格字符。
- 请记住检查配置文件中的 `dataset->transform_pipeline->SARLabelEncode->lower` 参数的值。如果词典中有大小写字母而且想区分大小写的话，请将其设置为 False。


## 参考文献
<!--- Guideline: Citation format GB/T 7714 is suggested. -->
[1] Xiaoyu Yue, Zhanghui Kuang, Chenhao Lin, Hongbin Sun, Wayne Zhang. RobustScanner: Dynamically Enhancing Positional Clues for Robust Text Recognition. arXiv:2007.07542, ECCV'2020
