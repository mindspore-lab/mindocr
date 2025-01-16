[English](README.md) | 中文

# MobileNetV3用于文字方向分类

## 概述

### MobileNetV3: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

MobileNetV3[[1](#参考文献)]于2019年发布，这个版本结合了V1的deep separable convolution，V2的Inverted Residuals and Linear Bottleneck，以及SE(Squeeze and Excitation)模块，并使用NAS（Neural Architecture Search）搜索最优网络的配置和参数。MobileNetV3 首先使用 MnasNet 进行粗粒度的结构搜索，然后使用强化学习从一组离散选择中选择最优配置。另外，MobileNetV3 还使用 NetAdapt 对架构进行微调。总之，MobileNetV3是一个轻量级的网络，在分类、检测和分割任务上有不错的表现。


<p align="center">
  <img src="https://user-images.githubusercontent.com/53842165/210044297-d658ca54-e6ff-4c0f-8080-88072814d8e6.png" width=800 />
</p>
<p align="center">
  <em>图 1. MobileNetV3整体架构图 [<a href="#参考文献">1</a>] </em>
</p>

### 文字方向分类器

在某些图片中，文字方向是反过来或不正确的，导致文字无法被正确识别。因此，我们使用了文字方向分类器来对文字方向进行分类并校正。MobileNetV3论文提出了两个版本的MobileNetV3：*MobileNetV3-Large*和*MobileNetV3-Small*。为了兼顾性能和分类准确性，我们采用*MobileNetV3-Small*作为文字方向分类器。

目前我们支持对0度和180度的文字方向进行分类。你也可以更新yaml配置文件中的`label_list`和`num_classes`参数来训练自己的文字方向分类器。0度和180度的图片示例如下。

<div align="center">

| **0度**         | **180度**    |
|-------------------|----------------|
|<img src="https://github.com/HaoyangLee/mindocr/assets/20376974/7dd4432f-775c-4a04-b9e7-0f52f68c70ee" width=200 /> | <img src="https://github.com/HaoyangLee/mindocr/assets/20376974/cfe298cd-08be-4866-b650-5eae560d59fa" width=200 /> |
|<img src="https://github.com/HaoyangLee/mindocr/assets/20376974/666da152-2e9b-48ae-b1a7-e190deef34d6" width=200 /> | <img src="https://github.com/HaoyangLee/mindocr/assets/20376974/802617fe-910e-451d-b202-e05da47b1196" width=200 /> |

</div>


## 实验结果

| mindspore |  ascend driver  |   firmware   | cann toolkit/kernel |
|:---------:|:---------------:|:------------:|:-------------------:|
|   2.3.1   |    24.1.RC2     | 7.3.0.1.231  |    8.0.RC2.beta1    |

MobileNetV3在ImageNet上预训练。另外，我们进一步在RCTW17、MTWI和LSVT数据集上进行了文字方向分类任务的训练。

在采用图模式的ascend 910*上实验结果，mindspore版本为2.3.1
<div align="center">

| **模型名称**    | **卡数** | **单卡批量大小** | **img/s** | **准确率** | **配置**               | **权重**                                                                                   |
|-------------|--------|------------|-----------|---------|----------------------|------------------------------------------------------------------------------------------|
| MobileNetV3 | 4      | 256        | 5923.5    | 94.59%  | [yaml](cls_mv3.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/cls/cls_mobilenetv3-92db9c58.ckpt) |
</div>


## 快速上手

### 安装

请参考MindOCR套件的[安装指南](https://github.com/mindspore-lab/mindocr#installation) 。

### 数据准备

#### ICDAR2015 数据集

请下载[RCTW17](https://rctw.vlrlab.net/dataset)、[MTWI](https://tianchi.aliyun.com/competition/entrance/231684/introduction)和[LSVT](https://rrc.cvc.uab.es/?ch=16&com=introduction)数据集，然后参考[数据转换](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters/README_CN.md)章节对数据集和标注进行格式转换。

完成数据准备工作后，数据的目录结构应该如下所示：

``` text
.
├── all_images
│   ├── img_1.jpg
│   ├── img_2.jpg
│   └── ...
├── eval_cls_gt.txt
├── train_cls_gt.txt
└── val_cls_gt.txt
```

> 用户如果想要使用自己的数据集进行训练，请参考[数据转换](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters/README_CN.md)对数据集和标注进行格式转换。


### 配置说明


在配置文件中更新数据集路径。其中`dataset_root`会分别和`data_dir`以及`label_file`拼接构成完整的数据集目录和标签文件路径。

```yaml
...
train:
  ckpt_save_dir: './tmp_cls'
  dataset:
    type: RecDataset
    dataset_root: dir/to/dataset          <--- 更新
    data_dir: all_images                  <--- 更新
    label_file: train_cls_gt.txt          <--- 更新
...
eval:
  dataset:
    type: RecDataset
    dataset_root: dir/to/dataset          <--- 更新
    data_dir: all_images                  <--- 更新
    label_file: val_cls_gt.txt            <--- 更新
...
```

> 【可选】可以根据CPU核的数量设置`num_workers`参数的值。



MobileNetV3由2个部分组成：`backbone`和`head`。具体来说:

```yaml
model:
  type: cls
  transform: null
  backbone:
    name: cls_mobilenet_v3_small_100
    pretrained: True  # 是否使用ImageNet上的预训练权重
  head:
    name: MobileNetV3Head
    out_channels: 1024  # arch=small 1024, arch=large 1280
    num_classes: *num_classes  # 2 or 4
```

### 评估

评估环节，在yaml配置文件中将`ckpt_load_path`参数配置为checkpoint文件的路径，并设置`distribute`为`False`，然后运行：

``` shell
python tools/eval.py -c configs/cls/mobilenetv3/cls_mv3.yaml
```

## 参考文献

<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Howard A, Sandler M, Chu G, et al. Searching for mobilenetv3[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2019: 1314-1324.
