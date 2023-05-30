[English](README.md) | 中文

# PSENet

<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> PSENet: [Shape Robust Text Detection With Progressive Scale Expansion Network](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Shape_Robust_Text_Detection_With_Progressive_Scale_Expansion_Network_CVPR_2019_paper.html)  

## 1. 概述

### PSENet

PSENet是一种基于语义分割的文本检测算法。它可以精确定位具有任意形状的文本实例，而大多数基于anchor类的算法不能用来检测任意形状的文本实例。此外，两个彼此靠近的文本可能会导致模型做出错误的预测。因此，为了解决上述问题，PSENet还提出了一种渐进式尺度扩展算法（Progressive Scale Expansion Algorithm, PSE）,利用该算法可以成功识别相邻的文本实例[[1](#参考文献)]。

<p align="center"><img alt="Figure 1. Overall PSENet architecture" src="https://github.com/VictorHe-1/mindocr_pse/assets/80800595/6ed1b691-52c4-4025-b256-a022aa5ef582" width="800"/></p>
<p align="center"><em>图 1. PSENet整体架构图</em></p>

PSENet的整体架构图如图1所示，包含以下阶段:

1. 使用Resnet作为骨干网络，从2，3，4，5阶段进行不同层级的特征提取；
2. 将提取到的特征放入FPN网络中，提取不同尺度的特征并拼接；
3. 将第2阶段的特征采用PSE算法生成最后的分割结果，并生成文本边界框。


## 2. 实验结果

### ICDAR2015
<div align="center">

| **模型**              | **环境配置**       | **骨干网络**      | **预训练数据集** | **Recall** | **Precision** | **F-score** | **训练时间**     | **吞吐量**   | **配置文件**                            | **模型权重下载**                                                                                                                                                                                                |
|---------------------|----------------|---------------|------------|------------|---------------|-------------|--------------|-----------|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PSENet               | D910x8-MS2.0-G | ResNet-152   | ImageNet   | 79.39%     | 84.91%        | 82.06%      | 138 s/epoch   | 7.57 img/s | [yaml](pse_r152_icdar15.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_resnet152_ic15-6058a798.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_resnet152_ic15-6058a798-0d755205.mindir) 
</div>

#### 注释：
- 环境配置：训练的环境配置表示为 {处理器}x{处理器数量}-{MS模式}，其中 Mindspore 模式可以是 G-graph 模式或 F-pynative 模式。
- PSENet的训练时长受数据处理部分超参和不同运行环境的影响非常大。

## 3. 快速上手

### 3.1 安装

请参考MindOCR套件的[安装指南](https://github.com/mindspore-lab/mindocr#installation) 。

### 3.2 数据准备

#### 3.2.1 ICDAR2015 数据集

请从[该网址](https://rrc.cvc.uab.es/?ch=4&com=downloads)下载ICDAR2015数据集，然后参考[数据转换](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters/README_CN.md)对数据集标注进行格式转换。

完成数据准备工作后，数据的目录结构应该如下所示： 

``` text
.
├── test
│   ├── images
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   └── ...
│   └── test_det_gt.txt
└── train
    ├── images
    │   ├── img_1.jpg
    │   ├── img_2.jpg
    │   └── ....jpg
    └── train_det_gt.txt
```

### 3.3 配置说明

在配置文件`configs/det/psenet/pse_r152_icdar15.yaml`中更新如下文件路径。其中`dataset_root`会分别和`data_dir`以及`label_file`拼接构成完整的数据集目录和标签文件路径。

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



PSENet由3个部分组成：`backbone`、`neck`和`head`。具体来说:

```yaml
model:
  type: det
  transform: null
  backbone:
    name: det_resnet152  
    pretrained: True    # 是否使用ImageNet数据集上的预训练权重
  neck:
    name: PSEFPN         # PSENet的特征金字塔网络
    out_channels: 128
  head:
    name: PSEHead
    hidden_size: 256
    out_channels: 7     # kernels数量
```

### 3.4 训练
* 后处理

训练前，请确保在/mindocr/postprocess/pse目录下按照以下方式编译后处理代码：

``` shell 
python3 setup.py build_ext --inplace
```

* 单卡训练

请确保yaml文件中的`distribute`参数为False。

``` shell 
# train psenet on ic15 dataset
python tools/train.py --config configs/det/psenet/pse_r152_icdar15.yaml
```

* 分布式训练

请确保yaml文件中的`distribute`参数为True。

```shell
# n is the number of GPUs/NPUs
mpirun --allow-run-as-root -n 8 python tools/train.py --config configs/det/psenet/pse_r152_icdar15.yaml
```

训练结果（包括checkpoint、每个epoch的性能和曲线图）将被保存在yaml配置文件的`ckpt_save_dir`参数配置的路径下，默认为`./tmp_det`。 

### 3.5 评估

评估环节，在yaml配置文件中将`ckpt_load_path`参数配置为checkpoint文件的路径，设置`distribute`为False，然后运行： 

``` shell
python tools/eval.py --config configs/det/psenet/pse_r152_icdar15.yaml
```

### 3.6 MindSpore Lite 推理

在进行推理前，请确保PSENet的后处理部分已编译（参考训练章节的后处理部分），并完成[推理环境搭建](../../../docs/cn/inference/environment_cn.md)。完成上述步骤后，请先下载[MindIR模型](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_resnet152_ic15-6058a798-0d755205.mindir)文件或使用以下命令将已训练完成的ckpt导出成MindIR文件:

``` shell
python tools/export.py --model_name psenet_resnet152 --data_shape 1472 2624 --local_ckpt_path /path/to/local_ckpt.ckpt
```
使用converter_lite工具将MindIR转换成MindSpore Lite支持的MindIR模型：
```shell
converter_lite \
    --saveType=MINDIR \
    --NoFusion=false \
    --fmk=MINDIR \
    --device=Ascend \
    --modelFile=psenet_resnet152.mindir \
    --outputFile=output \
    --configFile=config.txt
```
上述命令将生成output.om以及output.mindir模型文件。其中，config.txt文件配置如下：
```
 [ascend_context]
 input_format=NCHW
 input_shape=x:[1,3,1472,2624]
```

完成output.mindir文件导出后，在/mindocr/deploy/py_infer目录下使用以下命令即可进行推理和评估：
```shell
python infer.py \
    --input_images_dir=/your_path_to/test_images \
    --device=Ascend \
    --device_id=your_device_id \
    --parallel_num=2 \
    --precision_mode=fp32 \
    --det_model_path=your_path_to/output.mindir \
    --det_model_name=en_ms_det_psenet_resnet152 \
    --backend=lite \
    --save_log_dir=your_logs_dir \
    --res_save_dir=your_prediction_result_dir

python ../eval_utils/eval_det.py --gt_path=/your_path_to/det_gt.txt --pred_path=your_prediction_result_dir/det_results.txt
```
## 参考文献

<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Wang, Wenhai, et al. "Shape robust text detection with progressive scale expansion network." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.