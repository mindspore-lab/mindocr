[English](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/master/README.md) | 中文

# MASTER
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [MASTER: Multi-Aspect Non-local Network for Scene Text Recognition](https://arxiv.org/abs/1910.02562)

## 1. 模型描述
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->

基于注意力机制的场景文本识别器已经取得了巨大的成功，它利用仅占用更小中间表示的RNN编码器-解码器架构，来学习1维或2维的注意力。然而，这样的方法由于编码特征之间的相似度高，导致在基于RNN的局部注意力机制下出现了注意力失调问题。此外，基于RNN的方法由于并行化效率低而效率差。为了克服这些问题，本文提出了MASTER，一种基于自注意力机制的场景文本识别器，它(1)不仅编码了输入输出的注意力，还学习了Encoder和Decoder中的特征-特征和目标-目标关系，(2)学习了更强大和鲁棒的中间表示，以应对空间失真，(3)由于高度并行训练和高效的内存缓存机制，具有较高的训练效率和较快的推理速度。在各种基准测试中的广泛实验证明，MASTER在正常和不规则场景文本上表现出优异的性能。[<a href="#参考文献">1</a>]

<!--- Guideline: If an architecture table/figure is available in the paper, put one here and cite for intuitive illustration. -->

<p align="center">
  <img src="https://github.com/zhtmike/mindocr/assets/8342575/cd3121ca-e58f-4f45-b336-dc0134e0564e" width=450 />
</p>
<p align="center">
  <em> 图1. MASTER结构 [<a href="#参考文献">1</a>] </em>
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

### 精度结果

根据我们的实验，在公开基准数据集（IC03，IC13，IC15，IIIT，SVT，SVTP，CUTE）上的评估结果如下：

<div align="center">

| **模型** | **环境配置** | **平均准确率** | **训练时间** | **FPS** | **配置文件** | **模型权重下载** |
| :-----: | :-----:  | :-----: | :-----: | :-----: |:--------: | :-----: |
| Master-Resnet31     | D910x4-MS1.10-G | 90.20%    | 3721 s/epoch        | 4632 | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/master/master_resnet31.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/master/master_resnet31-7565c75f.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/master/master_resnet31_ascend-7565c75f-65015efe.mindir) |
</div>

<details open markdown>
  <div align="center">
  <summary>在各个基准数据集上的准确率</summary>

  | **模型** | **IC03_860** | **IC03_867** | **IC13_857** | **IC13_1015** | **IC15_1811** | **IC15_2077** | **IIIT5k_3000** | **SVT** | **SVTP** | **CUTE80** | **平均准确率** |
  | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
  | Master-ResNet31| 95.81% | 95.73%  | 96.97% | 95.57% | 81.83% | 78.29% | 96.33% | 90.57% | 82.33% | 88.54% | 90.20% |
  </div>
</details>

**注意:**
- 环境配置：训练的环境配置表示为 {处理器}x{处理器数量}-{MS模式}，其中 Mindspore 模式可以是 G-graph 模式或 F-pynative 模式。例如，D910x8-MS1.10-G 用于使用图形模式在8张昇腾910 NPU上依赖Mindspore1.10版本进行训练。
- 如需在其他环境配置重现训练结果，请确保全局批量大小与原配置文件保持一致。
- 模型都是从头开始训练的，无需任何预训练。关于训练和测试数据集的详细介绍，请参考[数据集下载及使用](#312-数据集下载)章节。
- Master的MindIR导出时的输入Shape均为(1, 3, 48, 160)。

## 3. 快速开始
### 3.1 环境及数据准备

#### 3.1.1 安装
环境安装教程请参考MindOCR的 [installation instruction](https://github.com/mindspore-lab/mindocr#installation).

#### 3.1.2 数据集准备

##### 3.1.2.1 MJSynth, 验证集和测试集
部分LMDB格式的训练及验证数据集可以从[这里](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0) (出处: [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here))下载。连接中的文件包含多个压缩文件，其中:
- `data_lmdb_release.zip` 包含了了部分数据集，有训练集(training/），验证集(validation/)以及测试集(evaluation)。
    - `training.zip` 包括两个数据集，分别是 [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/) 和 [SynthText (ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)。 这里我们只使用**MJSynth**。
    - `validation.zip` 是多个单独数据集的训练集的一个合集，包括[IC13](http://rrc.cvc.uab.es/?ch=2), [IC15](http://rrc.cvc.uab.es/?ch=4), [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), 和 [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)。
    - `evaluation.zip` 包含多个基准评估数据集，有[IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset), [IC03](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions), [IC13](http://rrc.cvc.uab.es/?ch=2), [IC15](http://rrc.cvc.uab.es/?ch=4), [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf)和 [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html)
- `validation.zip`: 与 data_lmdb_release.zip 中的validation/ 一样。
- `evaluation.zip`: 与 data_lmdb_release.zip 中的evaluation/ 一样。

##### 3.1.2.2 SynthText dataset

我们不使用`data_lmdb_release.zip`提供的`SynthText`数据, 因为它只包含部分切割下来的图片。请从<https://www.robots.ox.ac.uk/~vgg/data/scenetext/>下载原始数据, 并使用以下命令转换成LMDB格式

```bash
python tools/dataset_converters/convert.py \
    --dataset_name synthtext \
    --task rec_lmdb \
    --image_dir path_to_SynthText \
    --label_dir path_to_SynthText_gt.mat \
    --output_path ST_full
```
`ST_full` 包含了所有已切割的图片，以LMDB格式储存。 请将 `ST` 文件夹换成 `ST_full` 文件夹。

##### 3.1.2.3 SynthAdd dataset

另外请从<https://pan.baidu.com/s/1uV0LtoNmcxbO-0YA7Ch4dg>（密码：627x）下载**SynthAdd**训练集. 这个训练集是由<https://arxiv.org/abs/1811.00751>提出。请使用以下命令转换成LMDB格式

```bash
python tools/dataset_converters/convert.py \
    --dataset_name synthadd \
    --task rec_lmdb \
    --image_dir path_to_SynthAdd \
    --output_path SynthAdd
```

并将转换完成的`SynthAdd`文件夹摆在`/training`里面.

#### 3.1.3 数据集使用

最终数据文件夹结构如下：

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
│   ├── ST_full
│   │   ├── data.mdb
│   │   └── lock.mdb
│   └── SythAdd
│       ├── data.mdb
│       └── lock.mdb
└── validation
    ├── data.mdb
    └── lock.mdb
```

在这里，我们使用 `training/` 文件夹下的数据集进行训练，并使用联合数据集 `validation/` 进行验证。训练后，我们使用 `evaluation/` 下的数据集来评估模型的准确性。

**Training:** (total 17,402,659 samples)
- [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/)
  - Train: 21.2 GB, 7224586 samples
  - Valid: 2.36 GB, 802731 samples
  - Test: 2.61 GB, 891924 samples
- [SynthText (ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)
  - 17.0 GB, 7266529 samples
- [SynthAdd (SynthAdd)](https://arxiv.org/abs/1811.00751)
  - 2.7 GB, 1216889 samples

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
...
eval:
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # 验证数据集根目录
    data_dir: validation/                                             # 验证数据集目录，将与`dataset_root`拼接形成完整验证数据集目录
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
  ...
```

#### 3.1.4 检查配置文件
除了数据集的设置，请同时重点关注以下变量的配置：`system.distribute`, `system.val_while_train`, `common.batch_size`, `train.ckpt_save_dir`, `train.dataset.dataset_root`, `train.dataset.data_dir`, `train.dataset.label_file`,
`eval.ckpt_load_path`, `eval.dataset.dataset_root`, `eval.dataset.data_dir`, `eval.dataset.label_file`, `eval.loader.batch_size`。说明如下：

```yaml
system:
  distribute: True                                                    # 分布式训练为True，单卡训练为False
  amp_level: 'O2'
  amp_level_infer: "O2"
  seed: 42
  val_while_train: True                                               # 边训练边验证
  drop_overflow_update: False
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
- 由于全局批大小 （batch_size x num_devices） 是对结果复现很重要，因此当GPU/NPU卡数发生变化时，调整`batch_size`以保持全局批大小不变，或根据新的全局批大小线性调整学习率。


### 3.2 模型训练
<!--- Guideline: Avoid using shell script in the command line. Python script preferred. -->

* 分布式训练

使用预定义的训练配置可以轻松重现报告的结果。对于在多个昇腾910设备上的分布式训练，请将配置参数`distribute`修改为True，并运行：

```shell
# 在多个 GPU/Ascend 设备上进行分布式训练
mpirun --allow-run-as-root -n 8 python tools/train.py --config configs/rec/master/master_resnet31.yaml
```


* 单卡训练

如果要在没有分布式训练的情况下在较小的数据集上训练或微调模型，请将配置参数`distribute`修改为False 并运行：

```shell
# CPU/GPU/Ascend 设备上的单卡训练
python tools/train.py --config configs/rec/master/master_resnet31.yaml
```

训练结果（包括checkpoint、每个epoch的性能和曲线图）将被保存在yaml配置文件的`ckpt_save_dir`参数配置的目录下，默认为`./tmp_rec`。

### 3.3 模型评估

若要评估已训练模型的准确性，可以使用`eval.py`。请在yaml配置文件的`eval`部分将参数`ckpt_load_path`设置为模型checkpoint的文件路径，设置`distribute`为False，然后运行：

```shell
python tools/eval.py --config configs/rec/master/master_resnet31.yaml
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
- 请记住检查配置文件中的 `dataset->transform_pipeline->RecMasterLabelEncode->lower` 参数的值。如果词典中有大小写字母而且想区分大小写的话，请将其设置为 False。


## 5. MindSpore Lite 推理

请参考[MindOCR 推理](../../../docs/cn/inference/inference_tutorial.md)教程，基于MindSpore Lite在Ascend 310上进行模型的推理，包括以下步骤：

**1. 模型导出**

请先[下载](#2-评估结果)已导出的MindIR文件，或者参考[模型导出](../../README.md)教程，使用以下命令将训练完成的ckpt导出为MindIR文件:

```shell
python tools/export.py --model_name master_resnet31 --data_shape 48 160 --local_ckpt_path /path/to/local_ckpt.ckpt
# or
python tools/export.py --model_name configs/rec/master/master_resnet31.yaml --data_shape 48 160 --local_ckpt_path /path/to/local_ckpt.ckpt
```

其中，`data_shape`是导出MindIR时的模型输入Shape的height和width，下载链接中MindIR对应的shape值见[注释](#2-评估结果)。

**2. 环境搭建**

请参考[环境安装](../../../docs/cn/inference/environment.md#2-mindspore-lite推理)教程，配置MindSpore Lite推理运行环境。

**3. 模型转换**

请参考[模型转换](../../../docs/cn/inference/convert_tutorial.md#1-mindocr模型)教程，使用`converter_lite`工具对MindIR模型进行离线转换，
其中`configFile`文件中的`input_shape`需要填写模型导出时shape，如上述的(1, 3, 48, 160)，格式为NCHW。

**4. 执行推理**

假设在模型转换后得到output.mindir文件，在`deploy/py_infer`目录下使用以下命令进行推理：

```shell
python infer.py \
    --input_images_dir=/your_path_to/test_images \
    --device=Ascend \
    --device_id=0 \
    --rec_model_path=your_path_to/output.mindir \
    --rec_model_name_or_config=../../configs/rec/master/master_resnet31.yaml \
    --backend=lite \
    --res_save_dir=results_dir
```

## 参考文献
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Ning Lu, Wenwen Yu, Xianbiao Qi, Yihao Chen, Ping Gong, Rong Xiao, Xiang Bai. MASTER: Multi-Aspect Non-local Network for Scene Text Recognition. arXiv preprint arXiv:1910.02562, 2019.
