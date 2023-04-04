[English](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/crnn/README.md) | 中文

# CRNN
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition](https://https://arxiv.org/abs/1507.05717)

## 1. 模型描述
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->

卷积递归神经网络 (CRNN) 将 CNN 特征提取和 RNN 序列建模以及转录集成到一个统一的框架中。

如架构图（图 1）所示，CRNN 首先通过卷积层从输入图像中提取特征序列。由此一来，图像由提取的序列特征图表示，其中每个向量都与输入图像上的感受野相关联。 为了进一步处理特征，CRNN 采用循环神经网络层来预测每个帧的标签分布。为了将分布映射到文本字段，CRNN 添加了一个转录层，以将每帧预测转换为最终标签序列。 [<a href="#references">1</a>]

<!--- Guideline: If an architecture table/figure is available in the paper, put one here and cite for intuitive illustration. -->

<p align="center">
  <img src="https://user-images.githubusercontent.com/26082447/224601239-a569a1d4-4b29-4fa8-804b-6690cb50caef.PNG" width=450 />
</p>
<p align="center">
  <em> 图1. CRNN架构图 [<a href="#references">1</a>] </em>
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

根据我们的实验，在公开基准数据集（IC03，IC13，IC15，IIIT，SVT，SVTP，CUTE）上的评估结果如下：

<div align="center">

| **模型**           | **环境配置**       | **骨干网络**      | **平均准确率** | 训练时间(s/epoch) | **配置文件**                                                                          | **模型权重下载**                                                                             | 
|------------------|----------------|---------------|-----------|---------------|------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| CRNN (ours)      | D910x8-MS1.8-G | VGG7          | 82.03%    | 2445          | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/crnn/crnn_vgg7.yaml)     | [weights](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_vgg7-ea7e996c.ckpt)     |
| CRNN (ours)      | D910x8-MS1.8-G | ResNet34_vd   | 84.45%    | 2118          | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/crnn/crnn_resnet34.yaml) | [weights](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34-83f37f07.ckpt) |
| CRNN (PaddleOCR) | -              | ResNet34_vd   | 83.99%    | -             | -                                                                                              | -                                                                                          |

</div>

**注意:**
- 环境配置：训练的环境配置表示为 {处理器}x{处理器数量}-{MS模式}，其中 Mindspore 模式可以是 G-graph 模式或 F-pynative 模式。例如，D910x8-MS1.8-G 用于使用图形模式在8张昇腾910 NPU上依赖Mindspore1.8版本进行训练。
- VGG 和 ResNet 模型都是从头开始训练的，无需任何预训练。
- 上述模型是用 MJSynth(MJ)和 SynthText(ST)数据集训练的。更多数据详情，请参考[数据集准备](#312-数据集准备)章节。
- **评估在每个基准数据集上单独测试，平均准确度是所有子数据集的精度平均值。**
- PaddleOCR版CRNN，我们直接用的是其[github](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_rec_crnn_en.md)上面提供的已训练好的模型。


## 3. 快速开始
### 3.1 环境及数据准备

#### 3.1.1 安装
环境安装教程请参考MindOCR的 [installation instruction](https://github.com/mindspore-lab/mindocr#installation).

#### 3.1.2 数据集准备
LMDB格式的训练及验证数据集可以从[这里](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0) (出处: [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here))下载。连接中的文件包含多个压缩文件，其中:
- `data_lmdb_release.zip` 包含了**完整**的一套数据集，有训练集(training.zip），验证集(validation.zip)以及测试集(evaluation.zip)。
- `validation.zip` 是一个整合的验证集。
- `evaluation.zip` 包含多个基准评估数据集。

解压文件并完成数据准备操作后，数据文件夹结构如下：

``` text
.
├── training
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

#### 3.1.3 检查配置文件
请重点关注以下变量的配置：`system.distribute`, `system.val_while_train`, `common.batch_size`, `train.ckpt_save_dir`, `train.dataset.dataset_root`, `train.dataset.data_dir`, `train.dataset.label_file`, 
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
- 进行**模型训练**时，若`system.val_while_train`参数为`True`，将开启边训练边验证模式，此时需将`eval.dataset.dataset_root`设置为**验证**数据集根目录，将`eval.dataset.data_dir`设置为**验证**数据集目录（如`validation/`），将`eval.dataset.label_file`设置为**验证**数据集的标签文件路径（当数据集为LMDB格式时无需配置label_file）。
- 进行**模型评估**时，请将`eval.dataset.dataset_root`设置为**评估**数据集根目录，将`eval.dataset.data_dir`设置为**评估**数据集目录（如`evaluation/DATASET_NAME/`），将`eval.dataset.label_file`设置为**评估**数据集的标签文件路径（当数据集为LMDB格式时无需配置label_file）。

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

## 参考文献
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Baoguang Shi, Xiang Bai, Cong Yao. An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition. arXiv preprint arXiv:1507.05717, 2015.
