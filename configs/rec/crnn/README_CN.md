[English](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/crnn/README.md) | 中文

# CRNN
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)

## 1. 模型描述
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->

卷积递归神经网络 (CRNN) 将 CNN 特征提取和 RNN 序列建模以及转录集成到一个统一的框架中。

如架构图（图 1）所示，CRNN 首先通过卷积层从输入图像中提取特征序列。由此一来，图像由提取的序列特征图表示，其中每个向量都与输入图像上的感受野相关联。 为了进一步处理特征，CRNN 采用循环神经网络层来预测每个帧的标签分布。为了将分布映射到文本字段，CRNN 添加了一个转录层，以将每帧预测转换为最终标签序列。 [<a href="#参考文献">1</a>]

<!--- Guideline: If an architecture table/figure is available in the paper, put one here and cite for intuitive illustration. -->

<p align="center">
  <img src="https://user-images.githubusercontent.com/26082447/224601239-a569a1d4-4b29-4fa8-804b-6690cb50caef.PNG" width=450 />
</p>
<p align="center">
  <em> 图1. CRNN架构图 [<a href="#参考文献">1</a>] </em>
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

根据我们的实验，训练（[模型训练](#32-模型训练)）性能和精度评估（[模型评估](#33-模型评估)）结果如下：

<div align="center">

| **模型** | **环境配置** | **骨干网络** | **训练集** | **参数量** | **单卡批量** | **图模式8卡训练 (s/epoch)** | **图模式8卡训练 (ms/step)** | **图模式8卡训练 (FPS)** | **平均评估精度** | **配置文件** | **模型权重下载** |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| CRNN      | D910x8-MS1.8-G | VGG7 | MJ+ST | 8.72 M | 16 |  2488.82 | 22.06 | 5802.71 | 82.03%  | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/crnn/crnn_vgg7.yaml)     | [ckpt](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_vgg7-ea7e996c.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_vgg7-ea7e996c-573dbd61.mindir) |
| CRNN      | D910x8-MS1.8-G | ResNet34_vd | MJ+ST | 24.48 M | 64 |  2157.18 | 76.48 | 6694.84 | 84.45% | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/crnn/crnn_resnet34.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34-83f37f07.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34-83f37f07-eb10a0c9.mindir) |
</div>

- 在各个基准数据集（IC03，IC13，IC15，IIIT，SVT，SVTP，CUTE）上的准确率：

  <div align="center">

  | **模型** | **骨干网络** | **IC03_860** | **IC03_867** | **IC13_857** | **IC13_1015** | **IC15_1811** | **IC15_2077** | **IIIT5k_3000** | **SVT** | **SVTP** | **CUTE80** | **平均准确率** |
  | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
  | CRNN | VGG7 | 94.53% | 94.00% | 92.18% | 90.74% | 71.95% | 66.06% | 84.10% | 83.93% | 73.33% | 69.44% | 82.03% |
  | CRNN | ResNet34_vd | 94.42% | 94.23% | 93.35% | 92.02% | 75.92% | 70.15% | 87.73% | 86.40% | 76.28% | 73.96% | 84.45% |
  </div>


### 推理端

推理端的性能测试主要是基于Mindspore Lite，详细的操作介绍可参考 [Mindspore Lite推理](#6-mindspore-lite-推理)。

<div align="center">

| 设备 | 编译环境 | 模型 | 骨干网络 | 参数量 | 测试集 | 批量大小 | 图模式单卡推理 (FPS) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Ascend310P | Lite2.0 | CRNN | ResNet34_vd | 24.48 M | IC15 | 1 | 361.09 |
| Ascend310P | Lite2.0 | CRNN | ResNet34_vd | 24.48 M | SVT | 1 | 274.67 |

</div>

**注意:**
- 环境配置：训练的环境配置表示为 {处理器}x{处理器数量}-{MS模式}，其中 Mindspore 模式可以是 G-graph 模式或 F-pynative 模式。例如，D910x8-MS1.8-G 用于使用图形模式在8张昇腾910 NPU上依赖Mindspore1.8版本进行训练。
- 如需在其他环境配置重现训练结果，请确保全局批量大小与原配置文件保持一致。
- 模型所能识别的字符都是默认的设置，即所有英文小写字母a至z及数字0至9，详细请看[4. 字符词典](#4-字符词典)
- 模型都是从头开始训练的，无需任何预训练。关于训练和测试数据集的详细介绍，请参考[数据集下载及使用](#312-数据集下载)章节。
- CRNN_VGG7和CRNN_ResNet34_vd的MindIR导出时的输入Shape均为(1, 3, 32, 100)。


## 3. 快速开始
### 3.1 环境及数据准备

#### 3.1.1 安装
环境安装教程请参考MindOCR的 [installation instruction](https://github.com/mindspore-lab/mindocr#installation).

#### 3.1.2 数据集下载
LMDB格式的训练及验证数据集可以从[这里](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0) (出处: [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here))下载。连接中的文件包含多个压缩文件，其中:
- `data_lmdb_release.zip` 包含了**完整**的一套数据集，有训练集(training/），验证集(validation/)以及测试集(evaluation)。
    - `training.zip` 包括两个数据集，分别是 [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/) 和 [SynthText (ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)
    - `validation.zip` 是多个单独数据集的训练集的一个合集，包括[IC13](http://rrc.cvc.uab.es/?ch=2), [IC15](http://rrc.cvc.uab.es/?ch=4), [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), 和 [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)。
    - `evaluation.zip` 包含多个基准评估数据集，有[IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset), [IC03](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions), [IC13](http://rrc.cvc.uab.es/?ch=2), [IC15](http://rrc.cvc.uab.es/?ch=4), [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf)和 [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html)
- `validation.zip`: 与 data_lmdb_release.zip 中的validation/ 一样。
- `evaluation.zip`: 与 data_lmdb_release.zip 中的evaluation/ 一样。

#### 3.1.3 数据集使用

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

在这里，我们使用 `training/` 文件夹下的数据集进行 **training**，并使用联合数据集 `validation/` 进行验证。训练后，我们使用 `evaluation/` 下的数据集来评估模型的准确性。

**Training:** (total 14,442,049 samples)
- [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/)
  - Train: 21.2 GB, 7224586 samples
  - Valid: 2.36 GB, 802731 samples
  - Test: 2.61 GB, 891924 samples
- [SynthText (ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)
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

**模型训练的数据配置**

如欲重现模型的训练，建议修改配置yaml如下：

```yaml
...
train:
  ...
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # 训练数据集根目录
    data_dir: training/                                               # 训练数据集目录，将与`dataset_root`拼接形成完整训练数据集目录
    # label_files:                                                    # 训练数据集的标签文件路径，将与`dataset_root`拼接形成完整的训练数据的标签文件路径。当数据集为LMDB格式时无需配置
...
eval:
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # 验证数据集根目录
    data_dir: validation/                                             # 验证数据集目录，将与`dataset_root`拼接形成完整验证数据集目录
    # label_file:                                                     # 验证数据集的标签文件路径，将与`dataset_root`拼接形成完整的验证或评估数据的标签文件路径。当数据集为LMDB格式时无需配置
  ...
```

**模型评估的数据配置**

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
    # label_file:                                                     # 评估数据集的标签文件路径，将与`dataset_root`拼接形成完整的评估数据的标签文件路径。当数据集为LMDB格式时无需配置
  ...
```

通过使用上述配置 yaml 运行 [模型评估](#33-model-evaluation) 部分中所述的`tools/eval.py`，您可以获得数据集 CUTE80 的准确度性能。


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
    # label_file:                                                     # Path of evaluation label file, concatenated with `dataset_root` to be the complete path of evaluation label file, not required when using LMDBDataset
  ...
```

#### 3.1.4 检查配置文件
除了数据集的设置，请同时重点关注以下变量的配置：`system.distribute`, `system.val_while_train`, `common.batch_size`, `train.ckpt_save_dir`, `train.dataset.dataset_root`, `train.dataset.data_dir`, `train.dataset.label_file`,
`eval.ckpt_load_path`, `eval.dataset.dataset_root`, `eval.dataset.data_dir`, `eval.dataset.label_file`, `eval.loader.batch_size`。说明如下：

```yaml
system:
  distribute: True                                                    # 分布式训练为True，单卡训练为False
  amp_level: 'O3'
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
    dataset_root: dir/to/data_lmdb_release/                           # 训练数据集根目录
    data_dir: training/                                               # 训练数据集目录，将与`dataset_root`拼接形成完整训练数据集目录
    # label_files:                                                    # 训练数据集的标签文件路径，将与`dataset_root`拼接形成完整的训练数据的标签文件路径。当数据集为LMDB格式时无需配置
...
eval:
  ckpt_load_path: './tmp_rec/best.ckpt'                               # checkpoint文件路径
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # 验证或评估数据集根目录
    data_dir: validation/                                             # 验证或评估数据集目录，将与`dataset_root`拼接形成完整验证或评估数据集目录
    # label_file:                                                     # 验证或评估数据集的标签文件路径，将与`dataset_root`拼接形成完整的验证或评估数据的标签文件路径。当数据集为LMDB格式时无需配置
  ...
  loader:
      shuffle: False
      batch_size: 64                                                  # 验证或评估批大小
...
```

**注意:**
- 由于全局批大小 （batch_size x num_devices） 是对结果复现很重要，因此当GPU/NPU卡数发生变化时，调整`batch_size`以保持全局批大小不变，或将学习率线性调整为新的全局批大小。


### 3.2 模型训练
<!--- Guideline: Avoid using shell script in the command line. Python script preferred. -->

* 分布式训练

使用预定义的训练配置可以轻松重现报告的结果。对于在多个昇腾910设备上的分布式训练，请将配置参数`distribute`修改为True，并运行：

```shell
# 在多个 GPU/Ascend 设备上进行分布式训练
mpirun --allow-run-as-root -n 8 python tools/train.py --config configs/rec/crnn/crnn_resnet34.yaml
```


* 单卡训练

如果要在没有分布式训练的情况下在较小的数据集上训练或微调模型，请将配置参数`distribute`修改为False 并运行：

```shell
# CPU/GPU/Ascend 设备上的单卡训练
python tools/train.py --config configs/rec/crnn/crnn_resnet34.yaml
```

训练结果（包括checkpoint、每个epoch的性能和曲线图）将被保存在yaml配置文件的`ckpt_save_dir`参数配置的目录下，默认为`./tmp_rec`。

### 3.3 模型评估

若要评估已训练模型的准确性，可以使用`eval.py`。请在yaml配置文件的`eval`部分将参数`ckpt_load_path`设置为模型checkpoint的文件路径，设置`distribute`为False，然后运行：

```
python tools/eval.py --config configs/rec/crnn/crnn_resnet34.yaml
```

## 4. 字符词典

### 默认设置

在数据处理时，真实文本会根据提供的字符字典转换为标签 ID，字典中键是字符，值是 ID。默认情况下，字典 **"0123456789abcdefghijklmnopqrstuvwxyz"**，这代表着id=0 将对应字符'0'。在默认设置下，字典只考虑数字和小写英文字符，不包括空格。


### 内置词典

Mindocr内置了一部分字典，均放在了 `mindocr/utils/dict/` 位置，可选择合适的字典使用。

- `en_dict.txt` 是一个包含94个字符的英文字典，其中有数字，常用符号以及大小写的英文字母。
- `ch_dict.txt` 是一个包含6623个字符的中文字典，其中有常用的繁简体中文，数字，常用符号以及大小写的英文字母。


### 自定义词典

您也可以自定义一个字典文件 (***.txt)， 放在 `mindocr/utils/dict/` 下，词典文件格式应为每行一个字符的.txt 文件。


如需使用指定的词典，请将参数 `character_dict_path` 设置为字典的路径，并将参数 `num_classes` 改成对应的数量，即字典中字符的数量 + 1。


**注意：**
- 您可以通过将配置文件中的参数 `use_space_char` 设置为 True 来包含空格字符。
- 请记住检查配置文件中的 `dataset->transform_pipeline->RecCTCLabelEncode->lower` 参数的值。如果词典中有大小写字母而且想区分大小写的话，请将其设置为 False。


## 5. 中文识别模型训练

目前，CRNN模型支持中英文字识别并提供相应的预训练权重。详细内容如下

### 中文数据集准备及配置

我们采用公开的中文基准数据集[Benchmarking-Chinese-Text-Recognition](https://github.com/FudanVI/benchmarking-chinese-text-recognition)进行CRNN模型的训练和验证。

详细的数据准备和config文件配置方式, 请参考 [中文识别数据集准备](../../../docs/cn/datasets/chinese_text_recognition.md)

### 模型训练验证

准备好数据集和配置文件后，执行以下命令开启多卡训练

```shell
mpirun --allow-run-as-root -n 4 python tools/train.py --config configs/rec/crnn/crnn_resnet34_ch.yaml
```

### 评估结果和预训练权重
模型训练完成后，在测试集不同场景上的准确率评估结果如下。相应的模型配置和预训练权重可通过表中链接下载。

<div align="center">

| **模型** | **语种** | **环境配置** | **骨干网络** | **街景类** | **网页类** | **文档类** | **训练时间** | **FPS** | **配置文件** | **模型权重下载** |
| :-----: | :-----:  | :-------:  |:--------: | :--------: | :--------: | :--------: | :---------: |:--------: | :---------: | :-----------: |
| CRNN    | 中文 | D910x4-MS1.10-G | ResNet34_vd | 60.45% | 65.95% | 97.68% | 647 s/epoch | 1180 | [crnn_resnet34_ch.yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/crnn/crnn_resnet34_ch.yaml) |[ckpt](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34_ch-7a342e3c.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34_ch-7a342e3c-105bccb2.mindir) |
</div>

**注释：**

- MindIR导出时的输入Shape为(1, 3, 32, 320).


### 使用自定义数据集进行训练
您可以在自定义的数据集基于提供的预训练权重进行微调训练, 以在特定场景获得更高的识别准确率，具体步骤请参考文档 [使用自定义数据集训练识别网络](../../../docs/cn/tutorials/training_recognition_custom_dataset_CN.md)。


## 6. MindSpore Lite 推理

请参考[MindOCR 推理](../../../docs/cn/inference/inference_tutorial.md)教程，基于MindSpore Lite在Ascend 310上进行模型的推理，包括以下步骤：

**1. 模型导出**

请先[下载](#2-评估结果)已导出的MindIR文件，或者参考[模型导出](../../README.md)教程，使用以下命令将训练完成的ckpt导出为MindIR文件:

```shell
python tools/export.py --model_name crnn_resnet34 --data_shape 32 100 --local_ckpt_path /path/to/local_ckpt.ckpt
# or
python tools/export.py --model_name configs/rec/crnn/crnn_resnet34.yaml --data_shape 32 100 --local_ckpt_path /path/to/local_ckpt.ckpt
```

其中，`data_shape`是导出MindIR时的模型输入Shape的height和width，下载链接中MindIR对应的shape值见[注释](#2-评估结果)。

**2. 环境搭建**

请参考[环境安装](../../../docs/cn/inference/environment.md#2-mindspore-lite推理)教程，配置MindSpore Lite推理运行环境。

**3. 模型转换**

请参考[模型转换](../../../docs/cn/inference/convert_tutorial.md#1-mindocr模型)教程，使用`converter_lite`工具对MindIR模型进行离线转换，
其中`configFile`文件中的`input_shape`需要填写模型导出时shape，如上述的(1, 3, 32, 100)，格式为NCHW。

**4. 执行推理**

假设在模型转换后得到output.mindir文件，在`deploy/py_infer`目录下使用以下命令进行推理：

```shell
python infer.py \
    --input_images_dir=/your_path_to/test_images \
    --device=Ascend \
    --device_id=0 \
    --rec_model_path=your_path_to/output.mindir \
    --rec_model_name_or_config=../../configs/rec/crnn/crnn_resnet34.yaml \
    --backend=lite \
    --res_save_dir=results_dir
```


## 参考文献
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Baoguang Shi, Xiang Bai, Cong Yao. An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition. arXiv preprint arXiv:1507.05717, 2015.
