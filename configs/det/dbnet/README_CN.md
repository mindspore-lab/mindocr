[English](README.md) | 中文

# DBNet和DBNet++

<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> DBNet: [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)
> DBNet++: [Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion](https://arxiv.org/abs/2202.10304)

## 1. 概述

### DBNet

DBNet是一种基于分割的场景文本检测算法。在场景文本检测中，基于分割这类算法可以更加准确的描述各种形状的场景文本（比如弯曲形状的文本），而变得越来越流行。现有的基于分割的业界领先算法存在的缺陷是，概率图转化为文本框的二值化后处理过程通常需要人为设置一个阈值，而且后处理的聚合像素点的操作非常复杂且费时。

为了避免上述问题，DBNet在网络架构中集成了一个叫作“可微分二值化（Differentiable Binarization）”的自适应阈值。可微分二值化简化了后处理过程，增强了文本检测的性能。此外，在推理阶段移除该部分不会使性能降低[[1](#参考文献)]。

<p align="center"><img alt="Figure 1. Overall DBNet architecture" src="https://user-images.githubusercontent.com/16683750/225589619-d50c506c-e903-4f59-a316-8b62586c73a9.png" width="800"/></p>
<p align="center"><em>图 1. DBNet整体架构图</em></p>

DBNet的整体架构图如图1所示，包含以下阶段:

1. 使用Resnet-50作为骨干网络，从2，3，4，5阶段进行不同层级的特征提取；
2. 将提取到的特征放大，并以级联的方式与前一阶段提取到的特征加和；
3. 将第2阶段的特征再次放大到与最大的特征图相同的尺寸，并沿着通道轴连接。
4. 在最后的特征图（图中的深蓝色块）上应用3×3的卷积算子，和两个步长为2的去卷积算子来预测概率图和阈值图；
5. 通过可微分二值化将概率图和阈值图合并为一个近似二值图单元近似二值图，并生成文本边界框。

### DBNet++

DBNet++架构与DBNet相似，是DBNet的延伸。两者唯一的区别是，DBNet直接拼接从主干网络中提取和缩放的特征，而DBNet++使用一个自适应的模块（Adaptive Scale Fusion, ASF）来融合这些特征，如图2所示。
该模块可以自适应地融合不同尺寸的特征，有更好的尺寸（scale）鲁棒性。因此，DBNet++检测不同尺寸的文本的能力有显著提升。[[2](#参考文献)]

<p align="center"><img alt="Figure 2. Overall DBNet++ architecture" src="https://user-images.githubusercontent.com/16683750/236786997-13823b9c-ecaa-4bc5-8037-71299b3baffe.png" width="800"/></p>
<p align="center"><em>图 2. DBNet++整体架构图</em></p>

<p align="center"><img alt="Figure 3. Detailed architecture of the Adaptive Scale Fusion module" src="https://user-images.githubusercontent.com/16683750/236787093-c0c78d8f-e4f4-4c5e-8259-7120a14b0e31.png" width="700"/></p>
<p align="center"><em>图 3. Adaptive Scale Fusion模块架构图</em></p>

ASF由两个注意力模块组成——阶段注意力模块（stage-wise attention）和空间注意力模块（spatial attention），后者集成在前者中，如图3所示。
阶段注意模块学习不同尺寸的特征图的权重，而空间注意力模块学习跨空间维度的attention。这两个模块的组合使得模型可以获得尺寸（scale）鲁棒性很好的特征融合。
DBNet++在检测不同尺寸的文本方面表现更好，尤其是对于尺寸较大的文本；然而，DBNet在检测尺寸较大的文本时可能会生成不准确或分离的检测框。

## 2. 实验结果

DBNet和DBNet++在ICDAR2015，MSRA-TD500，SCUT-CTW1500，Total-Text和MLT2017数据集上训练。另外，我们在SynthText数据集上进行了预训练，并提供预训练权重下载链接。所有训练结果如下：

### ICDAR2015
<div align="center">

| **模型**              | **环境配置**       | **骨干网络**      | **预训练数据集** | **Recall** | **Precision** | **F-score** | **训练时间**     | **吞吐量**   | **配置文件**                            | **模型权重下载**                                                                                                                                                                                                |
|---------------------|----------------|---------------|------------|------------|---------------|-------------|--------------|-----------|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DBNet               | D910x1-MS2.0-G | MobileNetV3   | ImageNet       | 76.26%     | 78.22%        | 77.23%      | 10 s/epoch   | 100 img/s      | [yaml](db_mobilenetv3_icdar15.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_mobilenetv3-62c44539.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_mobilenetv3-62c44539-f14c6a13.mindir) |
| DBNet               | D910x1-MS2.0-G | ResNet-18     | ImageNet       | 80.12%     | 83.41%        | 81.73%      | 9.3 s/epoch  | 108 img/s      | [yaml](db_r18_icdar15.yaml)         | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18-0c0c4cfa.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18-0c0c4cfa-cf46eb8b.mindir)       |
| DBNet               | D910x1-MS2.0-G | ResNet-50     | ImageNet       | 83.53%     | 86.62%        | 85.05%      | 13.3 s/epoch | 75.2 img/s       | [yaml](db_r50_icdar15.yaml)         | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50-c3a4aa24.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50-c3a4aa24-fbf95c82.mindir)       |
|                     |                |               |            |            |               |             |              |           |                                     |                                                                                                                                                                                                           |
| DBNet++             | D910x1-MS2.0-G | ResNet-50     | SynthText  | 85.70%     | 87.81%        | 86.74%      | 17.7 s/epoch | 56 img/s  | [yaml](db++_r50_icdar15.yaml)       | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnetpp_resnet50-068166c2.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnetpp_resnet50-068166c2-9934aff0.mindir)   |

</div>

> 链接中模型DBNet的MindIR导出时的输入Shape为`(1,3,736,1280)`，模型DBNet++的MindIR导出时的输入Shape为`(1,3,1152,2048)`。

### MSRA-TD500

<div align="center">

| **模型**         | **环境配置**    | **骨干网络** | **预训练数据集** | **Recall** | **Precision** | **F-score** | **训练时间** | **吞吐量** | **配置文件**                  | **模型权重下载**                                                                                                                                                                                         |
|-------------------|----------------|--------------|----------------|------------|---------------|-------------|--------------|----------------|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DBNet            | D910x1-MS2.0-G | ResNet-18    | SynthText       | 79.55%     | 87.86%        | 83.50%      | 5.6 s/epoch  | 121.7 img/s      | [yaml](db_r18_td500.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18_td500-b5abff68.ckpt)  |
| DBNet            | D910x1-MS2.0-G | ResNet-50    | SynthText       | 83.68%     | 87.59%        | 85.59%      | 9.6 s/epoch  | 71.2 img/s      | [yaml](db_r50_td500.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_td500-0d12b5e8.ckpt)  |
</div>

> MSRA-TD500数据集有300训练集图片和200测试集图片，参考论文[Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)，我们训练此权重额外使用了来自HUST-TR400数据集的400训练集图片。可以在此下载全部[数据集](https://paddleocr.bj.bcebos.com/dataset/TD_TR.tar)用于训练。

### SCUT-CTW1500

<div align="center">

| **模型**         | **环境配置**    | **骨干网络** | **预训练数据集** | **Recall** | **Precision** | **F-score** | **训练时间** | **吞吐量** | **配置文件**                  | **模型权重下载**                                                                                                                                                                                         |
|-----------------|----------------|--------------|----------------|------------|---------------|-------------|--------------|----------------|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DBNet            | D910x1-MS2.0-G | ResNet-18    | SynthText       | 85.68%     | 85.33%        | 85.50%      | 8.2 s/epoch  | 122.1 img/s      | [yaml](db_r18_ctw1500.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18_ctw1500-0864b040.ckpt)  |
| DBNet            | D910x1-MS2.0-G | ResNet-50    | SynthText       | 86.72%     | 85.29%        | 86.00%      | 14.0 s/epoch  | 71.4 img/s      | [yaml](db_r50_ctw1500.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_ctw1500-f637e3d3.ckpt)  |
</div>

### Total-Text

<div align="center">

| **模型**         | **环境配置**    | **骨干网络** | **预训练数据集** | **Recall** | **Precision** | **F-score** | **训练时间** | **吞吐量** | **配置文件**                  | **模型权重下载**                                                                                                                                                                                         |
|-----------------|----------------|--------------|----------------|------------|---------------|-------------|--------------|----------------|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DBNet            | D910x1-MS2.0-G | ResNet-18    | SynthText       | 83.66%     | 87.65%        | 85.61%      | 12.9 s/epoch   |  96.9 img/s     | [yaml](db_r18_totaltext.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18_totaltext-fb456ff4.ckpt)  |
| DBNet            | D910x1-MS2.0-G | ResNet-50    | SynthText       | 84.79%     | 87.07%        | 85.91%      |  18.0 s/epoch  |   69.1 img/s     | [yaml](db_r50_totaltext.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_totaltext-76d6f421.ckpt)  |
</div>

### MLT2017

<div align="center">

| **模型**         | **环境配置**    | **骨干网络** | **预训练数据集** | **Recall** | **Precision** | **F-score** | **训练时间** | **吞吐量** | **配置文件**                  | **模型权重下载**                                                                                                                                                                                         |
|-----------------|---------------|-------------|----------------|------------|---------------|-------------|--------------|----------------|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DBNet           | D910x8-MS2.0-G | ResNet-18    | SynthText       | 72.55%     | 83.23%        | 77.52%      | 20.9 s/epoch  |  43.1 img/s      | [yaml](db_r18_mlt2017.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18_mlt2017-5af33809.ckpt)  |
| DBNet           | D910x8-MS2.0-G | ResNet-50    | SynthText      | 74.88%     | 83.77%        | 79.08%      | 23.6 s/epoch  |   38.2 img/s     | [yaml](db_r50_mlt2017.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_mlt2017-3bd6e569.ckpt)  |
</div>

### SynthText

<div align="center">

| **模型**         | **环境配置**    | **骨干网络** | **预训练数据集** | **训练Loss**| **训练时间** | **吞吐量** | **配置文件**                  | **模型权重下载**                 |
|-----------------|----------------|--------------|----------------|---------|---------|---------------|-------------|--------------|
| DBNet      | D910x1-MS2.0-G | ResNet-18    | ImageNet       |   2.41    |7075 s/epoch  | 121.37 img/s      | [yaml](db_r18_synthtext.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18_synthtext-251ef3dd.ckpt)  |
| DBNet      | D910x1-MS2.0-G | ResNet-50    | ImageNet       |   2.25    |10470 s/epoch  | 82.02 img/s      | [yaml](db_r50_synthtext.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_synthtext-40655acb.ckpt)  |
</div>

#### 注释：
- 环境配置：训练的环境配置表示为 {处理器}x{处理器数量}-{MS模式}，其中 Mindspore 模式可以是 G-graph 模式或 F-pynative 模式。
- DBNet的训练时长受数据处理部分和不同运行环境的影响非常大。


## 3. 快速上手

### 3.1 安装

请参考MindOCR套件的[安装指南](https://github.com/mindspore-lab/mindocr#installation) 。

### 3.2 数据准备

#### 3.2.1 ICDAR2015 数据集

请从[该网址](https://rrc.cvc.uab.es/?ch=4&com=downloads)下载ICDAR2015数据集，然后参考[数据转换](../../../tools/dataset_converters/README_CN.md)对数据集标注进行格式转换。

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

#### 3.2.2 MSRA-TD500 数据集

请从[该网址](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500))下载MSRA-TD500数据集，然后参考[数据转换](../../../tools/dataset_converters/README_CN.md)对数据集标注进行格式转换。

完成数据准备工作后，数据的目录结构应该如下所示：

```txt
MSRA-TD500
 ├── test
 │   ├── IMG_0059.gt
 │   ├── IMG_0059.JPG
 │   ├── IMG_0080.gt
 │   ├── IMG_0080.JPG
 │   ├── ...
 │   ├── train_det_gt.txt
 ├── train
 │   ├── IMG_0030.gt
 │   ├── IMG_0030.JPG
 │   ├── IMG_0063.gt
 │   ├── IMG_0063.JPG
 │   ├── ...
 │   ├── test_det_gt.txt
```

#### 3.2.3 SCUT-CTW1500 数据集

请从[该网址](https://github.com/Yuliang-Liu/Curve-Text-Detector)下载SCUT-CTW1500数据集，然后参考[数据转换](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters/README_CN.md)对数据集标注进行格式转换。

完成数据准备工作后，数据的目录结构应该如下所示：

```txt
ctw1500
 ├── test_images
 │   ├── 1001.jpg
 │   ├── 1002.jpg
 │   ├── ...
 ├── train_images
 │   ├── 0001.jpg
 │   ├── 0002.jpg
 │   ├── ...
 ├── test_det_gt.txt
 ├── train_det_gt.txt
```

#### 3.2.4 Total-Text 数据集

请从[该网址](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Dataset)下载Total-Text数据集，然后参考[数据转换](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters/README_CN.md)对数据集标注进行格式转换。

完成数据准备工作后，数据的目录结构应该如下所示：

```txt
totaltext
 ├── Images
 │   ├── Train
 │   │   ├── img1001.jpg
 │   │   ├── img1002.jpg
 │   │   ├── ...
 │   ├── Test
 │   │   ├── img1.jpg
 │   │   ├── img2.jpg
 │   │   ├── ...
 ├── test_det_gt.txt
 ├── train_det_gt.txt
```

#### 3.2.5 MLT2017 数据集
MLT2017数据集是一个多语言文本检测识别数据集，包含中文、日文、韩文、英文、法文、阿拉伯文、意大利文、德文和印度文共9种语言。请从[该网址](https://rrc.cvc.uab.es/?ch=8&com=downloads)下载MLT2017数据集，解压后请将数据中格式为.gif的图像转化为.jpg或.png格式。然后参考[数据转换](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters/README_CN.md)对数据集标注进行格式转换。

完成数据准备工作后，数据的目录结构应该如下所示：

```txt
MLT_2017
 ├── train
 │   ├── img_1.png
 │   ├── img_2.png
 │   ├── img_3.jpg
 │   ├── img_4.jpg
 │   ├── ...
 ├── validation
 │   ├── img_1.jpg
 │   ├── img_2.jpg
 │   ├── ...
 ├── train_det_gt.txt
 ├── validation_det_gt.txt
```
> 用户如果想要使用自己的数据集进行训练，请参考[数据转换](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters/README_CN.md)对数据集标注进行格式转换。并配置yaml文件，然后使用单卡或者多卡运行train.py进行训练即可，详细信息可参考下面几节教程。

#### 3.2.6 SynthText 数据集

请从[该网址](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c)下载SynthText数据集，解压后的数据的目录结构应该如下所示：

``` text
.
├── SynthText
│   ├── 1
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   └── ...
│   ├── 2
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   └── ...
│   ├── ...
│   ├── 200
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   └── ...
│   └── gt.mat

```

> :warning: 另外, 我们强烈建议在使用 `SynthText` 数据集之前先进行预处理，因为它包含一些错误的数据。可以使用下列的方式进行校正:
> ```shell
> python tools/dataset_converters/convert.py --dataset_name=synthtext --task=det --label_dir=/path-to-data-dir/SynthText/gt.mat --output_path=/path-to-data-dir/SynthText/gt_processed.mat
> ```
> 以上的操作会产生与`SynthText`原始标注格式相同但是是经过过滤后的标注数据.

### 3.3 配置说明

在配置文件`configs/det/dbnet/db_r50_icdar15.yaml`中更新如下文件路径。其中`dataset_root`会分别和`data_dir`以及`label_file`拼接构成完整的数据集目录和标签文件路径。

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

### 3.6 MindSpore Lite 推理

请参考[MindOCR 推理](../../../docs/cn/inference/inference_tutorial.md)教程，基于MindSpore Lite在Ascend 310上进行模型的推理，包括以下步骤：

- 模型导出

请先[下载](#2-实验结果)已导出的MindIR文件，或者参考[模型导出](../../README.md)教程，使用以下命令将训练完成的ckpt导出为MindIR文件:

```shell
python tools/export.py --model_name dbnet_resnet50 --data_shape 736 1280 --local_ckpt_path /path/to/local_ckpt.ckpt
# or
python tools/export.py --model_name configs/det/dbnet/db_r50_icdar15.yaml --data_shape 736 1280 --local_ckpt_path /path/to/local_ckpt.ckpt
```

其中，`data_shape`是导出MindIR时的模型输入Shape的height和width，下载链接中MindIR对应的shape值见[ICDAR2015注释](#ICDAR2015)。

- 环境搭建

请参考[环境安装](../../../docs/cn/inference/environment.md#2-mindspore-lite推理)教程，配置MindSpore Lite推理运行环境。

- 模型转换

请参考[模型转换](../../../docs/cn/inference/convert_tutorial.md#1-mindocr模型)教程，使用`converter_lite`工具对MindIR模型进行离线转换，
其中`configFile`文件中的`input_shape`需要填写模型导出时shape，如上述的(1,3,736,1280)，格式为NCHW。

- 执行推理


假设在模型转换后得到output.mindir文件，在`deploy/py_infer`目录下使用以下命令进行推理：

```shell
python infer.py \
    --input_images_dir=/your_path_to/test_images \
    --device=Ascend \
    --device_id=0 \
    --det_model_path=your_path_to/output.mindir \
    --det_model_name_or_config=../../configs/det/dbnet/db_r50_icdar15.yaml \
    --backend=lite \
    --res_save_dir=results_dir
```


## 参考文献

<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Minghui Liao, Zhaoyi Wan, Cong Yao, Kai Chen, Xiang Bai. Real-time Scene Text Detection with Differentiable
Binarization. arXiv:1911.08947, 2019

[2] Minghui Liao, Zhisheng Zou, Zhaoyi Wan, Cong Yao, Xiang Bai. Real-Time Scene Text Detection with Differentiable
Binarization and Adaptive Scale Fusion. arXiv:2202.10304, 2022
