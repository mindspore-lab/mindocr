
# ABINet
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [Read Like Humans: Autonomous, Bidirectional and Iterative Language
Modeling for Scene Text Recognition](https://arxiv.org/pdf/2103.06495)

## 1. 模型描述
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->
语义知识对场景文本识别有很大的帮助。然而，如何在端到端深度网络中有效地建模语义规则仍然是一个研究挑战。在本文中，我们认为语言模型的能力有限来自于:1)隐式语言建模;2)单向特征表示;3)带噪声输入的语言模型。相应地，我们提出了一种自主、双向、迭代的场景文本识别ABINet。首先，自主阻塞视觉和语言模型之间的梯度流，以强制显式语言建模。其次，提出了一种基于双向特征表示的新型双向完形填空式网络作为语言模型。第三，提出了一种语言模型迭代修正的执行方式，可以有效缓解噪声输入的影响。此外，我们提出了一种基于迭代预测集合的自训练方法，可以有效地从未标记的图像中学习。大量的实验表明，ABINet在低质量图像上具有优势，并在几个主流基准上取得了最先进的结果。此外，集成自训练训练的ABINet在实现人类水平的识别方面也有很大的进步 [<a href="#references">1</a>]

<!--- Guideline: If an architecture table/figure is available in the paper, put one here and cite for intuitive illustration. -->

<p align="center">
  <img src="https://github.com/safeandnewYH/ABINet_Figs/blob/main/images/ABINet_framework.png" width=1000 />
</p>
<p align="center">
  <em> 图1.  ABINet结构图 [<a href="#references">1</a>] </em>
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

### 精确度
根据我们的实验，在公共基准数据集(IC13、IC15、IIIT、SVT、SVTP、CUTE)上的评估结果如下:

<div align="center">

| **Model** | **Context** | **Avg Accuracy** | **Train T.** | **FPS** | **Recipe** | **Download** |
| :-----: | :-----------: | :--------------: | :----------: | :--------: | :--------: |:----------: |
| ABINet      | D910x8-MS1.9-G | 92.53%    | 22993 s/epoch       | 628.11 | [yaml](abinet_resnet45_en.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/abinet/abinet_resnet45_en-41e4bbd0.ckpt)
</div>

<details open>
  <div align="center">
  <summary>每个基准数据集的详细精度结果</summary>

  | **Model**  | **IC03_860** | **IC03_867** | **IC13_857** | **IC13_1015** | **IC15_1811** | **IC15_2077** | **IIIT5k_3000** | **SVT** | **SVTP** | **CUTE80** | **Average** |
  | :------:  | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
  | ABINet  | 95.81% | 96.08% | 97.32% | 95.47% | 86.31% | 82.52% | 96.37% | 93.82%| 89.61% | 92.01% | 92.53% |
  </div>
</details>




## 3. 快速开始
### 3.1 环境及数据准备

#### 3.1.1 安装
环境安装教程请参考MindOCR的 [installation instruction](https://github.com/mindspore-lab/mindocr#installation).


#### 3.1.2 Dataset Download
请下载LMDB数据集用于训练和评估
  - `training` 包含两个数据集: [MJSynth (MJ)](https://pan.baidu.com/s/1mgnTiyoR8f6Cm655rFI4HQ) 和 [SynthText (ST)](https://pan.baidu.com/s/1mgnTiyoR8f6Cm655rFI4HQ)
  - `evaluation` 包含几个基准数据集，它们是[IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset), [IC13](http://rrc.cvc.uab.es/?ch=2), [IC15](http://rrc.cvc.uab.es/?ch=4), [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf), 和 [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html).


数据结构应该被手动调整为

``` text
data_lmdb_release/
├── evaluation
│   ├── CUTE80
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC13_857
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC15_1811
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── ...
├── train
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
```

#### 3.1.3 数据集使用

在这里，我们使用 `train/` 文件夹下的数据集进行训练，我们使用 `evaluation/` 下的数据集来评估模型的准确性。

**Train:**  (total 15,895,356 samples)
- [MJSynth (MJ)](https://pan.baidu.com/s/1mgnTiyoR8f6Cm655rFI4HQ)
  - Train: 21.2 GB, 7224586 samples
  - Valid: 2.36 GB, 802731 samples
  - Test: 2.61 GB, 891924 samples
- [SynthText (ST)](https://pan.baidu.com/s/1mgnTiyoR8f6Cm655rFI4HQ)
  - Total: 24.6 GB, 6976115 samples


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
  ckpt_save_dir: './tmp_rec'                                          # 训练结果（包括checkpoint、每个epoch的性能和曲线图）保存目录
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # 训练数据集根目录
    data_dir: train/                                               # 训练数据集目录，将与`dataset_root`拼接形成完整训练数据集目录
    # label_files:                                                    # 训练数据集的标签文件路径，将与`dataset_root`拼接形成完整的训练数据的标签文件路径。当数据集为LMDB格式时无需配置
...
eval:
  ckpt_load_path: './tmp_rec/best.ckpt'                               # checkpoint文件路径
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # 验证或评估数据集根目录
    data_dir: evaluation/                                             # 验证或评估数据集目录，将与`dataset_root`拼接形成完整验证或评估数据集目录
    # label_file:                                                     # 验证或评估数据集的标签文件路径，将与`dataset_root`拼接形成完整的验证或评估数据的标签文件路径。当数据集为LMDB格式时无需配置
  ...
```

**模型评估的数据配置**
我们使用 `evaluation/` 下的数据集作为基准数据集。在**每个单独的数据集**（例如 CUTE80、IC03_860 等）上，我们通过将数据集的目录设置为评估数据集来执行完整评估。这样，我们就得到了每个数据集对应精度的列表，然后报告的精度是这些值的平均值。

如要重现报告的评估结果，您可以：
- 方法 1：对所有单个数据集重复评估步骤：CUTE80、IC13_857、IC15_1811、IIIT5k_3000、SVT、SVTP。然后取平均分。

- 方法 2：将所有基准数据集文件夹放在同一目录下，例如`evaluation/`。并使用脚本`tools/benchmarking/multi_dataset_eval.py`。

1.评估一个特定的数据集

例如，您可以通过修改配置 yaml 来评估数据集“CUTE80”上的模型，如下所示：

```yaml
...
train:
  # 不需要修改
...
eval:
  ckpt_load_path: './tmp_rec/best.ckpt'                               # checkpoint文件路径
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # 验证或评估数据集根目录
    data_dir: evaluation/CUTE80/                                             # 验证或评估数据集目录，将与`dataset_root`拼接形成完整验证或评估数据集目录
    # label_file:                                                     # 验证或评估数据集的标签文件路径，将与`dataset_root`拼接形成完整的验证或评估数据的标签文件路径。当数据集为LMDB格式时无需配置
  ...
```
通过使用上述配置 yaml 运行 [模型评估](#33-model-evaluation) 部分中所述的`tools/eval.py`，您可以获得数据集 CUTE80 的准确度性能。

2.对同一文件夹下的多个数据集进行评估

假设您已将所有 benckmark 数据集置于 evaluation/ 下，如下所示：

``` text
data_lmdb_release/
├── evaluation
│   ├── CUTE80
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC13_857
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC15_1811
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── ...
```

然后你可以通过如下修改配置yaml来评估每个数据集，并执行脚本`tools/benchmarking/multi_dataset_eval.py`。

```yaml
...
train:
  # 不需要修改
...
eval:
  ckpt_load_path: './tmp_rec/best.ckpt'                               # checkpoint文件路径
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # 验证或评估数据集根目录
    data_dir: evaluation/                                             # 验证或评估数据集目录，将与`dataset_root`拼接形成完整验证或评估数据集目录
    # label_file:                                                     # 验证或评估数据集的标签文件路径，将与`dataset_root`拼接形成完整的验证或评估数据的标签文件路径。当数据集为LMDB格式时无需配置
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
  batch_size: &batch_size 96                                          # 训练批大小
...
train:
  ckpt_save_dir: './tmp_rec'                                          # 训练结果（包括checkpoint、每个epoch的性能和曲线图）保存目录
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # 训练数据集根目录
    data_dir: train/                                               # 训练数据集目录，将与`dataset_root`拼接形成完整训练数据集目录
    # label_files:                                                    # 训练数据集的标签文件路径，将与`dataset_root`拼接形成完整的训练数据的标签文件路径。当数据集为LMDB格式时无需配置
...
eval:
  ckpt_load_path: './tmp_rec/best.ckpt'                               # checkpoint文件路径
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # 验证或评估数据集根目录
    data_dir: evaluation/                                             # 验证或评估数据集目录，将与`dataset_root`拼接形成完整验证或评估数据集目录
    # label_file:                                                     # 验证或评估数据集的标签文件路径，将与`dataset_root`拼接形成完整的验证或评估数据的标签文件路径。当数据集为LMDB格式时无需配置
  ...
  loader:
      shuffle: False
      batch_size: 96                                                  # 验证或评估批大小
...
```

**注意:**
- 由于全局批大小 （batch_size x num_devices） 是对结果复现很重要，因此当GPU/NPU卡数发生变化时，调整`batch_size`以保持全局批大小不变，或将学习率线性调整为新的全局批大小。
- 数据集:MJSynth和SynthText数据集来自作者公布的代码仓[ABINet_repo](https://github.com/FangShancheng/ABINet).


### 3.2 模型训练
<!--- Guideline: Avoid using shell script in the command line. Python script preferred. -->

* 分布式训练

使用预定义的训练配置可以轻松重现报告的结果。对于在多个昇腾910设备上的分布式训练，请将配置参数`distribute`修改为True，并运行：

```shell
# 在多个 GPU/Ascend 设备上进行分布式训练
mpirun --allow-run-as-root -n 8 python tools/train.py --config configs/rec/abinet/abinet_resnet45_en.yaml
```
ABINet模型训练时需要加载预训练模型，预训练模型的权重来自https://download.mindspore.cn/toolkits/mindocr/abinet/abinet_pretrain_en-821ca20b.ckpt，需要在“configs/rec/abinet/abinet_resnet45_en.yaml”中model的pretrained添加预训练权重的路径。

* 单卡训练

如果要在没有分布式训练的情况下在较小的数据集上训练或微调模型，请将配置参数`distribute`修改为False 并运行：

```shell
# CPU/GPU/Ascend 设备上的单卡训练
python tools/train.py --config configs/rec/abinet/abinet_resnet45_en.yaml
```



训练结果（包括checkpoint、每个epoch的性能和曲线图）将被保存在yaml配置文件的`ckpt_save_dir`参数配置的目录下，默认为`./tmp_rec`。

### 3.3 模型评估

若要评估已训练模型的准确性，可以使用`eval.py`。请在yaml配置文件的`eval`部分将参数`ckpt_load_path`设置为模型checkpoint的文件路径，设置`distribute`为False，然后运行：

```
python tools/eval.py --config configs/rec/abinet/abinet_resnet45_en.yaml
```

**注意:**
- 由于mindspore.nn.transformer在定义时需要固定的批处理大小，因此在选择val_while_train=True时，有必要确保验证集的批处理大小与模型的批处理大小相同。 所以tools/train.py中的第92行
```
refine_batch_size=True
```
应该被改为
```
refine_batch_size=False
```
- 同时， minocr.data.builder.py中的第179-185行
```
if not is_train:
    if drop_remainder and is_main_device:
        _logger.warning(
            "`drop_remainder` is forced to be False for evaluation "
            "to include the last batch for accurate evaluation."
        )
        drop_remainder = False

```
应该被改为
```
if not is_train:
    # if drop_remainder and is_main_device:
        _logger.warning(
            "`drop_remainder` is forced to be False for evaluation "
            "to include the last batch for accurate evaluation."
        )
        drop_remainder = True
```
## 参考文献
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Fang S, Xie H, Wang Y, et al. Read like humans: Autonomous, bidirectional and iterative language modeling for scene text recognition[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 7098-7107.
