[English](README.md) | 中文

# FCENet

<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> FCENet: [Fourier Contour Embedding for Arbitrary-Shaped Text Detection](https://arxiv.org/pdf/2104.10442.pdf)

## 1. 概述

### FCENet

FCENet是一种基于分割的场景文本检测算法。在场景文本检测中，基于分割这类算法可以更加准确的描述各种形状的场景文本（比如弯曲形状的文本），而变得越来越流行。
FCENet的一大亮点就是在任意不规则形状的文本场景上表现优异，这得益于它采用了可变形卷积[[1](#参考文献)]和傅里叶变换技术。
除此之外，FCENet还具有后处理简单和高泛化性的优点，在少量训练数据上训练就可以达到很好的效果。

#### 可变形卷积

可变形卷积的思想非常简单，就是将原来固定形状的卷积核变成可变的，在原始卷积的位置基础上，可变形卷积会产生一个随机方向的位置偏移，如下图所示：

<p align="center"><img alt="Figure 1" src="https://github.com/colawyee/mindocr/assets/15730439/20cdb21d-f0b4-4fdf-a8dd-5833d84648ba" width="600"/></p>
<p align="center"><em>图 1. 可变形卷积</em></p>

图(a)是原始的卷积核，图(b)是产生了随机方向位置偏移的可变形卷积核，图(c)(d)是图(b)的两种特殊情况。可以看出，这样做的好处是可以提升卷积核的几何变换能力，使其不仅局限于原始卷积核矩形的形状，而是可以支持更丰富的不规则形状。可变形卷积对不规则形状特征提取的效果会更好[[1](#参考文献)]，也更加适用于自然场景的文本识别场景。

#### 傅里叶轮廓线

傅里叶轮廓线是基于傅里叶变换的一种曲线拟合方法，随着傅里叶级数的项数k越大，就引入更多的高频信号，对轮廓刻画就越准确。下图展示了不同傅里叶级数情况下对不规则曲线的刻画能力：

<p align="center"><img width="445" alt="Image" src="https://github.com/colawyee/mindocr/assets/15730439/ae507f2f-ea4d-4787-90ea-4c59c634567a"></p>
<p align="center"><em>图 2. 傅里叶轮廓线渐进估计效果</em></p>

可以看出，随着傅里叶级数的项数k越大，其可以刻画的曲线是可以变得非常精细的。

#### 傅里叶轮廓线编码

傅里叶轮廓线编码是《Fourier Contour Embedding for Arbitrary-Shaped Text Detection》论文提出的一种将文本的轮廓的封闭曲线转化为一个向量（vector）的方法，也是FCENet算法需要用到的一种编码轮廓线的基础能力。傅里叶轮廓线编码方法通过在轮廓线上等间距的采样一些点，然后将采样的点的序列转化为傅里叶特征向量。值得注意的是，即使对于同一轮廓线，采样的点不同，对应生成的傅里叶特征向量也不相同。所以在采样的时候，需要限制起始点、间距和方向，保证对同一轮廓线生成的傅里叶特征向量的唯一性。

#### FCENet算法框架

<p align="center"><img width="800" alt="Image" src="https://github.com/colawyee/mindocr/assets/15730439/347aed5a-454c-4cfc-9577-fee163239626"></p>
<p align="center"><em>图 3. FCENet算法框架图</em></p>

像大多数OCR算法一样，FCENet的网络结构大体可以分为backbone，neck，head三个部分。其中backbone采用可变形卷积版本的Resnet50用于提取特征；neck部分采用特征金字塔[[2](#参考文献)]，特征金字塔是一组不同大小的卷积核，适用于提取原图中不同大小的特征，从而提高了目标检测的准确率，在一张图片中有不同大小的文本框的场景效果比较好；head部分有两条分支，一条是分类分支，用于预测文本区域和文本中心区域的热力图，通过比较该热力图与监督信号的交叉熵作为分类分支的损失值，另一条是回归分支，回归分支预测傅立叶特征向量，该向量用于通过傅立叶逆变换重构文本轮廓，通过计算重构文本轮廓线和监督信号的轮廓线在图像空间的smooth-l1 loss作为回归分支的损失值。


## 2. 实验结果

MindOCR中的FCENet网络在ICDAR 2015数据集上训练。训练结果如下：

### ICDAR2015
<div align="center">

| **模型**              | **环境配置**       | **骨干网络**      | **预训练数据集** | **Recall** | **Precision** | **F-score** | **训练时间**     | **吞吐量**   | **配置文件**                            | **模型权重下载**                                                                                                                                                                                                |
|---------------------|----------------|---------------|------------|------------|---------------|-------------|--------------|-----------|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FCENet               | D910x4-MS2.0-F | ResNet50   | ImageNet       | 81.5%     | 86.9%        | 84.1%      | 33 s/epoch   | 7 img/s      | [yaml](fce_icdar15.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/fcenet/) \| [mindir](https://download.mindspore.cn/toolkits/mindocr.mindir) |

</div>

#### 注释：
- 环境配置：训练的环境配置表示为 {处理器}x{处理器数量}-{MS模式}，其中 Mindspore 模式可以是 G-graph 模式或 F-pynative 模式。
- FCENet的训练时长受数据处理部分和不同运行环境的影响非常大。


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


> 用户如果想要使用自己的数据集进行训练，请参考[数据转换](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters/README_CN.md)对数据集标注进行格式转换。并配置yaml文件，然后使用单卡或者多卡运行train.py进行训练即可，详细信息可参考下面几节教程。

### 3.3 配置说明

在配置文件`configs/det/fcenet/fce_icdar15.yaml`中更新如下文件路径。其中`dataset_root`会分别和`data_dir`以及`label_file`拼接构成完整的数据集目录和标签文件路径。

```yaml
...
train:
  ckpt_save_dir: './tmp_det_fcenet'
  dataset_sink_mode: False
  ema: True
  dataset:
    type: DetDataset
    dataset_root: dir/to/dataset          <--- 更新
    data_dir: icda/ch4_training_images    <--- 更新
    label_file: icda/train_det_gt.txt     <--- 更新
...
eval:
  ckpt_load_path: '/best.ckpt'            <--- 更新
  dataset_sink_mode: False
  dataset:
    type: DetDataset
    dataset_root: dir/to/dataset          <--- 更新
    data_dir: icda/ch4_test_images        <--- 更新
    label_file: icda/test_det_gt.txt      <--- 更新
...
```

> 【可选】可以根据CPU核的数量设置`num_workers`参数的值。



FCENet由3个部分组成：`backbone`、`neck`和`head`。具体来说:

```yaml
model:
  resume: False
  type: det
  transform: null
  backbone:
    name: det_resnet50  # 暂时只支持ResNet50
    pretrained: True    # 是否使用ImageNet数据集上的预训练权重
  neck:
    name: FCEFPN        # FCENet的特征金字塔网络
    out_channel: 256
  head:
    name: FCEHead
    scales: [ 8, 16, 32 ]
    alpha: 1.2
    beta: 1.0
    fourier_degree: 5
    num_sample: 50
```

### 3.4 训练

* 单卡训练

请确保yaml文件中的`distribute`参数为False。

``` shell
# train fcenet on ic15 dataset
python tools/train.py --config configs/det/fcenet/fce_icdar15.yaml
```

* 分布式训练

请确保yaml文件中的`distribute`参数为True。

```shell
# n is the number of GPUs/NPUs
mpirun --allow-run-as-root -n 2 python tools/train.py --config configs/det/fcenet/fce_icdar15.yaml
```

训练结果（包括checkpoint、每个epoch的性能和曲线图）将被保存在yaml配置文件的`ckpt_save_dir`参数配置的路径下，默认为`./tmp_det`。

### 3.5 评估

评估环节，在yaml配置文件中将`ckpt_load_path`参数配置为checkpoint文件的路径，设置`distribute`为False，然后运行：

``` shell
python tools/eval.py --config configs/det/fcenet/fce_icdar15.yaml
```

### 3.6 MindSpore Lite 推理

请参考[MindOCR 推理](../../../docs/cn/inference/inference_tutorial_cn.md)教程，基于MindSpore Lite在Ascend 310上进行模型的推理，包括以下步骤：

- 模型导出

请先[下载](#2-实验结果)已导出的MindIR文件，或者参考[模型导出](../../README.md)教程，使用以下命令将训练完成的ckpt导出为MindIR文件:

```shell
python tools/export.py --model_name fcenet_resnet50 --data_shape 736 1280 --local_ckpt_path /path/to/local_ckpt.ckpt
# or
python tools/export.py --model_name configs/det/fcenet/db_r50_icdar15.yaml --data_shape 736 1280 --local_ckpt_path /path/to/local_ckpt.ckpt
```

其中，`data_shape`是导出MindIR时的模型输入Shape的height和width，下载链接中MindIR对应的shape值见[ICDAR2015注释](#ICDAR2015)。

- 环境搭建

请参考[环境安装](../../../docs/cn/inference/environment_cn.md#2-mindspore-lite推理)教程，配置MindSpore Lite推理运行环境。

- 模型转换

请参考[模型转换](../../../docs/cn/inference/convert_tutorial_cn.md#1-mindocr模型)教程，使用`converter_lite`工具对MindIR模型进行离线转换，
其中`configFile`文件中的`input_shape`需要填写模型导出时shape，如上述的(1,3,736,1280)，格式为NCHW。

- 执行推理

假设在模型转换后得到output.mindir文件，在`deploy/py_infer`目录下使用以下命令进行推理：

```shell
python infer.py \
    --input_images_dir=/your_path_to/test_images \
    --device=Ascend \
    --device_id=0 \
    --det_model_path=your_path_to/output.mindir \
    --det_model_name_or_config=../../configs/det/fcenet/fce_icdar15.yaml \
    --backend=lite \
    --res_save_dir=results_dir
```


## 参考文献

<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Dai, J., Qi, H., Xiong, Y., Li, Y., Zhang, G., Hu, H., & Wei, Y. (2017). Deformable Convolutional Networks. 2017 IEEE International Conference on Computer Vision (ICCV), 764-773.

[2] T. Lin, et al., "Feature Pyramid Networks for Object Detection," in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, USA, 2017 pp. 936-944. doi: 10.1109/CVPR.2017.106
