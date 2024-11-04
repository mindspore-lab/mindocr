# DBNet

<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> DBNet: [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)

## 1. 模型描述

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

该DBNet-PPOCRv3网络参考自PP-OCRv3 [<a href="#参考文献">2</a>] 的检测模块。其中针对DBNet的优化主要有：
 - LK-PAN：大感受野的PAN结构；
 - DML：教师模型互学习策略；
 - RSE-FPN：残差注意力机制的FPN结构；

## 2. 权重转换

如您已经有采用PaddleOCR训练好的PaddlePaddle模型，想在MindOCR下直接进行推理或进行微调续训，您可以将训练好的模型转换为MindSpore格式的ckpt文件。

运行param_converter.py脚本，输入需要进行转换的pdparams文件、权重名字对应关系json文件和ckpt输出路径，即可进行权重转换。

其中，权重名字对应关系json文件所包含的key和value分别为MindSpore参数名称和Paddle参数名称。

```shell
python tools/param_converter.py \
    -iuput_path path/to/paddleocr.pdparams \
    -json_path configs/det/dbnet/db_mobilenetv3_ppocrv3_param_map.json \
    -output_path path/to/output.ckpt
```



## 3. 模型训练
### 3.1 环境及数据准备

#### 3.1.1 安装
环境安装教程请参考MindOCR的 [installation instruction](https://github.com/mindspore-lab/mindocr#installation).

#### 3.1.2 数据集准备


目前，MindOCR检测网络支持两种输入格式，分别是:

-  `Common Dataset`：一种文件格式，存储图像、文本边界框和文本标注。目标文件格式的一个示例是：
``` text
img_1.jpg\t[{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]
```
它由 [DetDataset](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/data/det_dataset.py) 读取。如果您的数据集不是与示例格式相同的格式，请参阅 [说明](../datasets/converters.md) ，了解如何将不同数据集的注释转换为支持的格式。

- `SynthTextDataset`：由 [SynthText800k](https://github.com/ankush-me/SynthText) 提供的一种文件格式。 更多关于这个数据集的细节可以参考[这里](../datasets/synthtext.md)。它的标注文件是一个`.mat`文件，其中包括 `imnames`（图像名称）、`wordBB`（单词级边界框）、`charBB`（字符级边界框）和 `txt`（文本字符串）。它由 [SynthTextDataset](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/data/det_dataset.py) 读取。用户可以参考 `SynthTextDataset `来编写自定义数据集类。

我们建议用户将文本检测数据集准备成 `Common Dataset `格式，然后使用 `DetDataset` 来加载数据。以下教程进一步解释了详细步骤。


**训练集准备**

请将所有训练图像放在一个文件夹中，并在更高级别的目录中指定一个 txt 文件 `train_det.txt` ，来标记所有训练图像名称和对应的标签。txt 文件的一个示例如下：
```text
# 文件名	# 一个字典列表
img_1.jpg\t[{"transcription": "Genaxis Theatre", "points": [[377, 117], [463, 117], [465, 130], [378, 130]]}, {"transcription": "[06]", "points": [[493, 115], [519, 115], [519, 131], [493, 131]]}, {...}]
img_2.jpg\t[{"transcription": "guardian", "points": [[642, 250], [769, 230], [775, 255], [648, 275]]}]
...
```

*注意*：请使用 `\tab` 分隔图像名称和标签，避免使用空格或其他分隔符。

最终的训练集将以以下格式存储：

```
|-data
    |- train_det.txt
    |- training
        |- img_1.jpg
        |- img_2.jpg
        |- img_3.jpg
        | ...
```

**验证集准备**

类似地，请将所有测试图像放在一个文件夹中，并在更高级别的目录中指定一个 txt 文件 `val_det.txt` ，来标记所有测试图像名称和对应的标签。最终，测试集的文件夹将会以以下格式存储：
```
|-data
    |- val_det.txt
    |- validation
        |- img_1.jpg
        |- img_2.jpg
        |- img_3.jpg
        | ...
```

**模型训练的数据配置**

以`Common Dataset `为例，请修改配置文件对您的数据路径进行设置。

```yaml
...
train:
  ...
  dataset:
    type: DetDataset                                                  # 文件读取方法。这里我们使用 `Common Dataset` 格式
    dataset_root: dir/to/data/                                        # 数据的根目录
    data_dir: training/                                               # 训练数据集目录。它将与 `dataset_root` 拼接成一个完整的路径。
    label_file: train_det.txt                                         # 训练标签的路径。它将与 `dataset_root` 拼接成一个完整的路径。
...
eval:
  dataset:
    type: DetDataset                                                  # 文件读取方法。这里我们使用 `Common Dataset` 格式
    dataset_root: dir/to/data/                                        # 数据的根目录
    data_dir: validation/                                             # 测试数据集目录。它将与 `dataset_root` 拼接成一个完整的路径。
    label_file: val_det.txt                                           # 测试标签的路径。它将与 `dataset_root` 拼接成一个完整的路径。
  ...
```

#### 3.1.3 配置文件准备

以 `configs/det/dbnet/db_mobilenetv3_ppocrv3.yaml` 中的 `train.dataset.transform_pipeline` 为例。它指定了一组应用于图像或标签的转换函数，用以生成作为模型输入或损失函数输入的数据。这些转换函数定义在 `mindocr/data/transforms` 中。

```yaml
...
train:
...
  dataset:
        transform_pipeline:
      - DecodeImage:
          img_mode: RGB
          to_float32: False
      - DetLabelEncode:
      - RandomColorAdjust:
          brightness: 0.1255  # 32.0 / 255
          saturation: 0.5
      - RandomHorizontalFlip:
          p: 0.5
      - RandomRotate:
          degrees: [ -10, 10 ]
          expand_canvas: False
          p: 1.0
      - RandomScale:
          scale_range: [ 0.5, 3.0 ]
          p: 1.0
      - RandomCropWithBBox:
          max_tries: 10
          min_crop_ratio: 0.1
          crop_size: [ 960, 960 ]
          p: 1.0
      - ValidatePolygons:
      - ShrinkBinaryMap:
          min_text_size: 8
          shrink_ratio: 0.4
      - BorderMap:
          shrink_ratio: 0.4
          thresh_min: 0.3
          thresh_max: 0.7
      - NormalizeImage:
          bgr_to_rgb: False
          is_hwc: True
          mean: imagenet
          std: imagenet
      - ToCHWImage:
  ...
```

- `DecodeImage` 和 `DetLabelEncode`：这两个转换函数解析 `train_det.txt` 文件中的字符串，加载图像和标签，并将它们保存为一个字典；

- `RandomColorAdjust`， `RandomHorizontalFlip`， `RandomRotate`， `RandomScale` 和 `RandomCropWithBBox`：这些转换函数执行典型的图像增强操作。除了 `RandomColorAdjust`以外，其他函数都会改变边界框标签。

- `ValidatePolygons`：过滤掉由于之前的数据增强而在出现图像外部的边界框；

- `ShrinkBinaryMap `和 `BorderMap`：它们生成 `dbnet` 训练所需的二进制图和边界图

- `NormalizeImage`：它根据 `ImageNet` 数据集的均值和方差对图像进行归一化；

- `ToCHWImage`：它将 `HWC` 图像转换为 `CHW` 图像。

对于测试转换函数，所有的图像增强操作都被移除，被替换为一个简单的缩放函数。

```yaml
...
eval:
...
  dataset:
    transform_pipeline:
      - DecodeImage:
          img_mode: RGB
          to_float32: False
      - DetLabelEncode:
      - DetResize:
          limit_type: 'min'
          limit_side_len: 736
      - NormalizeImage:
          bgr_to_rgb: True
          is_hwc: True
          mean: imagenet
          std: imagenet
      - ToCHWImage:
...
```

#### 3.1.4 配置模型架构

虽然不同的模型有不同的架构，但 `MindOCR` 将它们形式化为一个通用的三阶段架构：`[backbone]->[neck]->[head]`。以 `configs/det/dbnet/db_mobilenetv3_ppocrv3.yaml` 为例：
```yaml
model:
  type: det
  transform: null
  backbone:
    name: det_mobilenet_v3_enhance
    architecture: large
    alpha: 0.5
    disable_se: True
    pretrained: False
  neck:
    name: RSEFPN
    out_channels: 96
    shortcut: True
  head:
    name: DBHeadEnhance
    k: 50
    bias: False
    adaptive: True
```

`backbone`,`neck`和`head`定义在 `mindocr/models/backbones`、`mindocr/models/necks` 和 `mindocr/models/heads` 下。

#### 3.1.5 配置训练超参数

`configs/det/dbnet/db_mobilenetv3_ppocrv3.yaml` 中定义了一些训练超参数，如下：
```yaml
metric:
  name: DetMetric
  main_indicator: f-score

loss:
  name: DBLoss
  eps: 1.0e-6
  l1_scale: 10
  bce_scale: 5
  bce_replace: diceloss

scheduler:
  scheduler: warmup_cosine_decay
  lr: 0.001
  min_lr: 0.0
  num_epochs: 500
  warmup_epochs: 2
  decay_epochs: 498

optimizer:
  opt: Adam
  beta1: 0.9
  beta2: 0.999
  weight_decay: 5.0e-05
```
它使用 `Adam` 优化器（在 `mindocr/optim/optim.factory.py` 中）和 `warmup_cosine_decay`（在 `mindocr/scheduler/scheduler_factory.py` 中）作为学习率调整策略。损失函数是 `DBLoss`（在 `mindocr/losses/det_loss.py` 中），评估指标是 `DetMetric`（在 `mindocr/metrics/det_metrics.py` 中）。



### 3.2 模型训练


用户可以使用我们提供的预训练模型做模型做为起始训练，预训练模型往往能提升模型的收敛速度甚至精度。以中文模型为例，我们提供的预训练模型网址为<https://download-mindspore.osinfra.cn/toolkits/mindocr/dbnet/dbnet_mobilenetv3_ppocrv3-70d6018f.ckpt>, 用户仅需在配置文件里添加`model.pretrained`添加对应网址如下：

```yaml
...
model:
  type: det
  transform: null
  backbone:
    name: det_mobilenet_v3_enhance
    architecture: large
    alpha: 0.5
    disable_se: True
    pretrained: False
  neck:
    name: RSEFPN
    out_channels: 96
    shortcut: True
  head:
    name: DBHeadEnhance
    k: 50
    bias: False
    adaptive: True
  pretrained: https://download-mindspore.osinfra.cn/toolkits/mindocr/dbnet/dbnet_mobilenetv3_ppocrv3-70d6018f.ckpt
...
```

如果遇到网络问题，用户可尝试预先把预训练模型下载到本地，把`model.pretained`改为本地地址如下：
```yaml
...
model:
  type: det
  transform: null
  backbone:
    name: det_mobilenet_v3_enhance
    architecture: large
    alpha: 0.5
    disable_se: True
    pretrained: False
  neck:
    name: RSEFPN
    out_channels: 96
    shortcut: True
  head:
    name: DBHeadEnhance
    k: 50
    bias: False
    adaptive: True
  pretrained: path/to/dbnet_mobilenetv3_ppocrv3-70d6018f.ckpt
...
```

如果用户不需要使用预训练模型，只需把`model.pretrained`删除即可。

* 分布式训练

在大量数据的情况下，建议用户使用分布式训练。对于在多个昇腾910设备的分布式训练，请将配置参数`system.distribute`修改为True, 例如：

```shell
# 在多个 Ascend 设备上进行分布式训练
mpirun --allow-run-as-root -n 4 python tools/train.py --config configs/det/dbnet/db_mobilenetv3_ppocrv3.yaml
```

* 单卡训练

如果要在没有分布式训练的情况下在较小的数据集上训练模型，请将配置参数`distribute`修改为False 并运行：

```shell
# CPU/Ascend 设备上的单卡训练
python tools/train.py --config configs/det/dbnet/db_mobilenetv3_ppocrv3.yaml
```

训练结果（包括checkpoint、每个epoch的性能和曲线图）将被保存在yaml配置文件的`ckpt_save_dir`参数配置的目录下，默认为`./tmp_rec`。


* 断点续训

如果用户期望在开始训练时同时加载模型的优化器，学习率等信息，并继续训练，可以在配置文件里面添加`model.resume`为对应的本地模型地址如下，并启动训练

```yaml
...
model:
  type: det
  transform: null
  backbone:
    name: det_mobilenet_v3_enhance
    architecture: large
    alpha: 0.5
    disable_se: True
    pretrained: False
  neck:
    name: RSEFPN
    out_channels: 96
    shortcut: True
  head:
    name: DBHeadEnhance
    k: 50
    bias: False
    adaptive: True
  resume: path/to/model.ckpt
...
```


### 3.3 模型评估

若要评估已训练模型的准确性，可以使用`tools/eval.py`。请在yaml配置文件的`eval`部分将参数`ckpt_load_path`设置为模型checkpoint的文件路径，设置`distribute`为`False`如下：

```yaml
system:
  distribute: False             # During evaluation stage, set to False
...
eval:
  ckpt_load_path: path/to/model.ckpt
```

然后运行

```shell
python tools/eval.py --config configs/det/dbnet/db_mobilenetv3_ppocrv3.yaml
```

为了在准确率和召回率中取得更好的平衡，您可能需要根据您的需求自行调整后处理中`box_thresh`的阈值，即推理结果置信度低于`box_thresh`的检测框会被过滤掉
```yaml
...
postprocess:
  name: DBPostprocess
  box_type: quad                # whether to output a polygon or a box
  binary_thresh: 0.3            # binarization threshold
  box_thresh: 0.9               # box score threshold 0.9           <--- 修改此处
  max_candidates: 1000
  expand_ratio: 1.5             # coefficient for expanding predictions
...
```

### 3.4 模型推理

用户可以通过使用推理脚本快速得到模型的推理结果。请先将图片放至在--image_dir指定的同一文件夹内，然后执行

```shell
python tools/infer/text/predict_det.py --image_dir {dir_to_your_image_data} --det_algorithm DB_PPOCRv3 --draw_img_save_dir inference_results
```
结果默认会存放于`./inference_results` 目录下。

## 参考文献

<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Minghui Liao, Zhaoyi Wan, Cong Yao, Kai Chen, Xiang Bai. Real-time Scene Text Detection with Differentiable
Binarization. arXiv:1911.08947, 2019

[2] PaddleOCR PP-OCRv3 https://github.com/PaddlePaddle/PaddleOCR/blob/344b7594e49f0fbb4d6578bd347505664ed728bf/doc/doc_ch/PP-OCRv3_introduction.md#2
