[English](README.md) | 中文

# EAST

<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155)

## 1. 概述

EAST (Efficient and Accurate Scene Text Detection)是一种高效、准确且轻量级的OCR检测算法，主要用于在自然场景下的文本检测。该算法使用深度残差网络提取文本特征，在特征金字塔网络中进行特征融合，并采用二分类和定位两个分支来检测文本。EAST在文本检测的准确性和鲁棒性方面取得了显著的成果。

<p align="center"><img alt="Figure 1. east_architecture" src="https://github.com/tonytonglt/mindocr/assets/54050944/4781c9aa-64a5-4963-bf02-6620d173dc9a" width="384"/></p>
<p align="center"><em>图 1. EAST整体架构图（我们使用ResNet50取代图中的PVANet）</em></p>

EAST的整体架构图如图1所示，包含以下阶段:

1.**特征提取**:
使用Resnet-50作为骨干网络，从2，3，4，5阶段进行不同层级的特征提取；

2.**特征融合**:
采用特征特征融合的方式，将骨干网络中不同层级的特征进行放大，并和更大的特征图沿通道轴进行连接，如此反复。使得模型可以对不同大小的文本区域进行处理，并提高检测的精度。

3.**边界框回归**:
对文本框的位置以及旋转角进行回归，使得EAST能够检测倾斜文本，完成自然场景下文本检测的任务。目前支持检测旋转矩形文本区域的文本框。

4.**文本分支**:
在确定了文本区域的位置和大小后，EAST模型会进一步将这些区域分类为文本或非文本区域。为此，模型采用了一条全卷积的文本分支，对文本区域进行二分类。

## 2. 实验结果

### ICDAR2015
<div align="center">

| **模型**           | **环境配置**       | **骨干网络**    | **预训练数据集** | **Recall** | **Precision** | **F-score** | **训练时间**    | **吞吐量**   | **配置文件**                   | 模型权重下载                                                                                                     |
|------------------|----------------|-------------|------------|------------|---------------|-------------|-------------|-----------|----------------------------|------------------------------------------------------------------------------------------------------------|
| EAST             | D910x8-MS1.9-G | ResNet-50   | ImageNet   | 82.23%     | 87.68%        | 84.87%      | 1.6 s/epoch | 625 img/s | [yaml](east_r50_icdar15.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/east/east_resnet50_ic15-7262e359.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/east/east_resnet50_ic15-7262e359-5f05cd42.mindir) |


</div>

#### 注释：
- 环境配置：训练的环境配置表示为 {处理器}x{处理器数量}-{MS模式}，其中 Mindspore 模式可以是 G-graph 模式或 F-pynative 模式。
- EAST的训练时长受数据处理部分和不同运行环境的影响非常大。

## 3. 快速上手

### 3.1 安装

请参考MindOCR套件的[安装指南](https://github.com/mindspore-lab/mindocr#installation) 。

### 3.2 数据准备

请从[该网址](https://rrc.cvc.uab.es/?ch=4&com=downloads)下载ICDAR2015数据集，然后参考[数据转换](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters/README_CN.md)对数据集标注进行格式转换。

完成数据准备工作后，数据的目录结构应该如下所示： 

``` text
.
├── test
│   ├── images
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   └── ...
│   └── test_det_gt.txt
└── train
    ├── images
    │   ├── img_1.jpg
    │   ├── img_2.jpg
    │   └── ....jpg
    └── train_det_gt.txt
```

### 3.3 配置说明

在配置文件`configs/det/east/east_r50_icdar15.yaml`中更新如下文件路径。其中`dataset_root`会分别和`dataset_root`以及`label_file`拼接构成完整的数据集目录和标签文件路径。

```yaml
...
train:
  ckpt_save_dir: './tmp_det'
  dataset_sink_mode: False
  dataset:
    type: DetDataset
    dataset_root: dir/to/dataset          <--- 更新
    data_dir: train/images                <--- 更新
    label_file: train/train_det_gt.txt    <--- 更新
...
eval:
  dataset_sink_mode: False
  dataset:
    type: DetDataset
    dataset_root: dir/to/dataset          <--- 更新
    data_dir: test/images                 <--- 更新
    label_file: test/test_det_gt.txt      <--- 更新
...
```

> 【可选】可以根据CPU核的数量设置`num_workers`参数的值。



EAST由3个部分组成：`backbone`、`neck`和`head`。具体来说:

```yaml
model:
  type: det
  transform: null
  backbone:
    name: det_resnet50
    pretrained: True    # 是否使用ImageNet数据集上的预训练权重
  neck:
    name: EASTFPN       # EAST的特征金字塔网络
    out_channels: 128
  head:
    name: EASTHead
```

### 3.4 训练

* 单卡训练

请确保yaml文件中的`distribute`参数为False。

``` shell 
# train east on ic15 dataset
python tools/train.py --config configs/det/east/east_r50_icdar15.yaml
```

* 分布式训练

请确保yaml文件中的`distribute`参数为True。

```shell
# n is the number of GPUs/NPUs
mpirun --allow-run-as-root -n 8 python tools/train.py --config configs/det/east/east_r50_icdar15.yaml
```

训练结果（包括checkpoint、每个epoch的性能和曲线图）将被保存在yaml配置文件的`ckpt_save_dir`参数配置的路径下，默认为`./tmp_det`。 

### 3.5 评估

评估环节，在yaml配置文件中将`ckpt_load_path`参数配置为checkpoint文件的路径，设置`distribute`为False，然后运行： 

``` shell
python tools/eval.py --config configs/det/east/east_r50_icdar15.yaml
```

## 参考文献

<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Xinyu Zhou, Cong Yao, He Wen, Yuzhi Wang, Shuchang Zhou, Weiran He, Jiajun Liang. EAST: An Efficient and Accurate Scene Text Detector. arXiv:1704.03155, 2017
