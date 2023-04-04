[English](README.md) | 中文

# DBNet

<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)

## 1. 概述

DBNet是一种基于分割的场景文本检测算法。在场景文本检测中，基于分割这类算法可以更加准确的描述各种形状的场景文本（比如弯曲形状的文本），而变得越来越流行。现有的基于分割的业界领先算法存在的缺陷是，概率图转化为文本框的二值化后处理过程通常需要人为设置一个阈值，而且后处理的聚合像素点的操作非常复杂且费时。

为了避免上述问题，DBNet在网络架构中集成了一个叫作“可微分二值化（Differentiable Binarization）”的自适应阈值。可微分二值化简化了后处理过程，增强了文本检测的性能。此外，在推理阶段移除该部分不会使性能降低[[1](#references)]。

![dbnet_architecture](https://user-images.githubusercontent.com/16683750/225589619-d50c506c-e903-4f59-a316-8b62586c73a9.png)
<p align="center"><em>图 1. DBNet整体架构图</em></p>

DBNet的整体架构图如图1所示，包含以下阶段:

1. 使用Resnet-50作为骨干网络，从2，3，4，5阶段进行不同层级的特征提取；
2. 将提取到的特征放大，并以级联的方式与前一阶段提取到的特征加和；
3. 将第2阶段的特征再次放大到与最大的特征图相同的尺寸，并沿着通道轴连接。
4. 在最后的特征图（图中的深蓝色块）上应用3×3的卷积算子，和两个步长为2的去卷积算子来预测概率图和阈值图；
5. 通过可微分二值化将概率图和阈值图合并为一个近似二值图单元近似二值图，并生成文本边界框。

## 2. 实验结果

### ICDAR2015
<div align="center">

| **模型**            | **环境配置**       | **骨干网络**      | **预训练数据集**  | **Recall**  | **Precision** | **F-score** | **训练时间(s/epoch)** | **配置文件**                    | **模型权重下载**                                                                                   |
|-------------------|----------------|---------------|-------------|-------------|---------------|-------------|-------------------|-----------------------------|----------------------------------------------------------------------------------------------|
| DBNet (ours)      | D910x1-MS1.9-G | ResNet-50     | ImageNet    | 81.70%      | 85.84%        | 83.72%      | 35                | [yaml](db_r50_icdar15.yaml) | [weights](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50-db1df47a.ckpt) |
| DBNet (PaddleOCR) | -              | ResNet50_vd   | SynthText   | 78.72%      | 86.41%        | 82.38%      | -                 | -                           | -                                                                                            |
| DBNet++           | D910x1-MS1.9-G | ResNet-50     | ImageNet    | 82.02%      | 87.38%        | 84.62%      | -                 | -                           | -                                                                                            |

</div>

> DBNet++的详细信息即将发布，敬请期待。DBNet和DBNet++的唯一区别在于_Adaptive Scale Fusion_模块, 在yaml配置文件`neck`模块中的 `use_asf`参数进行设置。

#### 注释：
- 环境配置：训练的环境配置表示为 {处理器}x{处理器数量}-{MS模式}，其中 Mindspore 模式可以是 G-graph 模式或 F-pynative 模式。
- DBNet的训练时长受数据处理部分和不同运行环境的影响非常大。

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

在配置文件`configs/det/dbnet/db_r50_icdar15.yaml`中更新如下文件路径。其中`dataset_root`会分别和`dataset_root`以及`label_file`拼接构成完整的数据集目录和标签文件路径。

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



DBNet由3个部分组成：`backbone`、`neck`和`head`。具体来说:

```yaml
model:
  type: det
  transform: null
  backbone:
    name: det_resnet50  # 暂时只支持ResNet50
    pretrained: True    # 是否使用ImageNet数据集上的预训练权重
  neck:
    name: DBFPN         # DBNet的特征金字塔网络
    out_channels: 256
    bias: False
    use_asf: False      # DBNet++的自适应尺度融合模块 (仅供DBNet++使用)
  head:
    name: DBHead
    k: 50               # 可微分二值化的放大因子
    bias: False
    adaptive: True      # 训练时设置为True, 推理时设置为False
```

[comment]: <> (_DBNet_和_DBNet++的唯一区别在于_Adaptive Scale Fusion_ module, 在`neck`模块中的 `use_asf`参数进行设置。)

### 3.4 训练

* 单卡训练

请确保yaml文件中的`distribute`参数为False。

``` shell 
# train dbnet on ic15 dataset
python tools/train.py --config configs/det/dbnet/db_r50_icdar15.yaml
```

* 分布式训练

请确保yaml文件中的`distribute`参数为True。

```shell
# n is the number of GPUs/NPUs
mpirun --allow-run-as-root -n 2 python tools/train.py --config configs/det/dbnet/db_r50_icdar15.yaml
```

训练结果（包括checkpoint、每个epoch的性能和曲线图）将被保存在yaml配置文件的`ckpt_save_dir`参数配置的路径下，默认为`./tmp_det`。 

### 3.5 评估

评估环节，在yaml配置文件中将`ckpt_load_path`参数配置为checkpoint文件的路径，设置`distribute`为False，然后运行： 

``` shell
python tools/eval.py --config configs/det/dbnet/db_r50_icdar15.yaml
```

## 参考文献

<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Minghui Liao, Zhaoyi Wan, Cong Yao, Kai Chen, Xiang Bai. Real-time Scene Text Detection with Differentiable Binarization. arXiv:1911.08947, 2019
