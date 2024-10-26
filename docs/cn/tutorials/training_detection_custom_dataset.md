# 使用自定义数据集训练检测网络

本文档提供了如何使用自定义数据集训练文本检测网络的教程。

- [使用自定义数据集训练检测网络](#使用自定义数据集训练检测网络)
  - [1. 数据集准备](#1-数据集准备)
    - [1.1 准备训练数据](#11-准备训练数据)
    - [1.2 准备测试数据](#12-准备测试数据)
  - [2. 配置文件准备](#2-配置文件准备)
    - [2.1 配置训练/测试数据集](#21-配置训练测试数据集)
    - [2.2 配置训练/测试转换函数](#22-配置训练测试转换函数)
    - [2.3 配置模型架构](#23-配置模型架构)
    - [2.4 配置训练超参数](#24-配置训练超参数)
  - [3. 模型训练, 测试和推理](#3-模型训练-测试和推理)
    - [3.1 训练](#31-训练)
    - [3.2 评估](#32-评估)
    - [3.3 推理](#33-推理)
      - [3.3.1 环境准备](#331-环境准备)
      - [3.3.2 模型转换](#332-模型转换)
      - [3.3.3 推理 (Python)](#333-推理-python)

## 1. 数据集准备
目前，MindOCR检测网络支持两种输入格式，分别是:

-  `Common Dataset`：一种文件格式，存储图像、文本边界框和文本标注。目标文件格式的一个示例是：
``` text
img_1.jpg\t[{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]
```
它由 [DetDataset](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/data/det_dataset.py) 读取。如果您的数据集不是与示例格式相同的格式，请参阅 [说明](../datasets/converters.md) ，了解如何将不同数据集的注释转换为支持的格式。

- `SynthTextDataset`：由 [SynthText800k](https://github.com/ankush-me/SynthText) 提供的一种文件格式。 更多关于这个数据集的细节可以参考[这里](../datasets/synthtext.md)。它的标注文件是一个`.mat`文件，其中包括 `imnames`（图像名称）、`wordBB`（单词级边界框）、`charBB`（字符级边界框）和 `txt`（文本字符串）。它由 [SynthTextDataset](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/data/det_dataset.py) 读取。用户可以参考 `SynthTextDataset `来编写自定义数据集类。

我们建议用户将文本检测数据集准备成 `Common Dataset `格式，然后使用 `DetDataset` 来加载数据。以下教程进一步解释了详细步骤。

### 1.1 准备训练数据

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

### 1.2 准备测试数据

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

## 2. 配置文件准备

为了准备相应的配置文件，用户应该指定训练和测试数据集的目录。

### 2.1 配置训练/测试数据集

请选择 `configs/det/dbnet/db_r50_icdar15.yaml` 作为初始配置文件，并修改其中的` train.dataset` 和 `eval.dataset` 字段。

```yaml
...
train:
  ...
  dataset:
    type: DetDataset                                                  # 文件读取方法。这里我们使用 `Common Dataset` 格式
    dataset_root: dir/to/data/                                        # 数据的根目录
    data_dir: training/                                               # 训练数据集目录。它将与 `dataset_root` 拼接成一个完整的路径。
    label_file: train_det.txt                                       # 训练标签的路径。它将与 `dataset_root` 拼接成一个完整的路径。
...
eval:
  dataset:
    type: DetDataset                                                  # 文件读取方法。这里我们使用 `Common Dataset` 格式
    dataset_root: dir/to/data/                                        # 数据的根目录
    data_dir: validation/                                             # 测试数据集目录。它将与 `dataset_root` 拼接成一个完整的路径。
    label_file: val_det.txt                                     # 测试标签的路径。它将与 `dataset_root` 拼接成一个完整的路径。
  ...
```
### 2.2 配置训练/测试转换函数

以 `configs/det/dbnet/dbnet_r50_icdar15.yaml` 中的 `train.dataset.transform_pipeline` 为例。它指定了一组应用于图像或标签的转换函数，用以生成作为模型输入或损失函数输入的数据。这些转换函数定义在 `mindocr/data/transforms` 中。


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
          crop_size: [ 640, 640 ]
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

- `ValidatePolygons`：它过滤掉由于之前的数据增强而在出现图像外部的边界框；

- `ShrinkBinaryMap `和 `BorderMap`：它们生成 `dbnet` 训练所需的二进制图和边界图

- `NormalizeImage`：它根据 `ImageNet` 数据集的均值和方差对图像进行归一化；

- `ToCHWImage`：它将 `HWC` 图像转换为 `CHW` 图像。

对于测试转换函数，所有的图像增强操作都被移除，被替换为一个简单的缩放函数。


```yaml
eval:
  dataset
    transform_pipeline:
      - DecodeImage:
          img_mode: RGB
          to_float32: False
      - DetLabelEncode:
      - DetResize:
          target_size: [ 736, 1280 ]
          keep_ratio: False
          force_divisable: True
      - NormalizeImage:
          bgr_to_rgb: False
          is_hwc: True
          mean: imagenet
          std: imagenet
      - ToCHWImage:
```

更多关于转换函数的教程可以在 [转换教程](transform_tutorial.md) 中找到。

### 2.3 配置模型架构

虽然不同的模型有不同的架构，但 `MindOCR` 将它们形式化为一个通用的三阶段架构：`[backbone]->[neck]->[head]`。以 `configs/det/dbnet/dbnet_r50_icdar15.yaml` 为例：

```yaml
model:
  type: det
  transform: null
  backbone:
    name: det_resnet50  # 目前只支持 ResNet50
    pretrained: True    # 是否使用在 ImageNet 上预训练的权重
  neck:
    name: DBFPN         # DBNet 的 FPN 部分
    out_channels: 256
    bias: False
    use_asf: False      # DBNet++ 中的自适应尺度融合模块（仅用于 DBNet++）
  head:
    name: DBHead
    k: 50               # 可微分二值化的放大因子
    bias: False
    adaptive: True      # 训练时为 True，推理时为 False
```

`backbone`,`neck`和`head`定义在 `mindocr/models/backbones`、`mindocr/models/necks` 和 `mindocr/models/heads` 下。

### 2.4 配置训练超参数

`configs/det/dbnet/dbnet_r50_icdar15.yaml` 中定义了一些训练超参数，如下所示：
```yaml
metric:
  name: DetMetric
  main_indicator: f-score

loss:
  name: DBLoss
  eps: 1.0e-6
  l1_scale: 10
  bce_scale: 5
  bce_replace: bceloss

scheduler:
  scheduler: polynomial_decay
  lr: 0.007
  num_epochs: 1200
  decay_rate: 0.9
  warmup_epochs: 3

optimizer:
  opt: SGD
  filter_bias_and_bn: false
  momentum: 0.9
  weight_decay: 1.0e-4
```
它使用 `SGD` 优化器（在 `mindocr/optim/optim.factory.py` 中）和 `polynomial_decay`（在 `mindocr/scheduler/scheduler_factory.py` 中）作为学习率调整策略。损失函数是 `DBLoss`（在 `mindocr/losses/det_loss.py` 中），评估指标是 `DetMetric`（在 `mindocr/metrics/det_metrics.py` 中）。

## 3. 模型训练, 测试和推理

当所有配置文件都已设置好后，用户就可以开始训练他们的模型。MindOCR支持在模型训练完成后进行测试和推理。

### 3.1 训练

* 单机训练

在单机训练中，模型是在单个设备上训练的（默认为`device:0`）。用户应该在`yaml`配置文件中将`system.distribute`设置为`False`。如果用户想要在除device:0以外的设备上运行这个模型，还需要将`system.device_id`设置为目标设备id。

以`configs/det/dbnet/db_r50_icdar15.yaml`为例，训练命令是：
```Shell
python tools/train.py -c=configs/det/dbnet/db_r50_icdar15.yaml
```

* 分布式训练

在分布式训练中，yaml配置文件中的`system.distribute`应该为`True`。在GPU和Ascend设备上，用户可以使用`mpirun`来启动分布式训练。例如，使用`device:0`和`device:1`进行训练：

```Shell
# n是GPU/NPU的数量
mpirun --allow-run-as-root -n 2 python tools/train.py --config configs/det/dbnet/db_r50_icdar15.yaml
```
有时，用户可能想要指定设备id来进行分布式训练，例如，`device:2`和`device:3`。

在GPU设备上，在运行上面的`mpirun`命令之前，用户可以运行以下命令：
```
export CUDA_VISIBLE_DEVICES=2,3
```
在Ascend设备上，用户应该创建一个像这样的`rank_table.json`：
```json
Copy{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "10.155.111.140",
            "device": [
                {"device_id": "2","device_ip": "192.3.27.6","rank_id": "2"},
                {"device_id": "3","device_ip": "192.4.27.6","rank_id": "3"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}

```

目标设备的`device_ip`可以通过运行`cat /etc/hccn.conf`获取。输出结果中的`address_x`就是`ip`地址。更多细节可以在[分布式训练教程](distribute_train.md)中找到。

### 3.2 评估

为了评估训练模型的准确性，用户可以使用`tools/eval.py`。

以单机评估为例。在yaml配置文件中，`system.distribute`应该为`False`；`eval.ckpt_load_path`应该是目标checkpoint路径；`eval.dataset_root`，`eval.data_dir`和`eval.label_file`应该指定为正确的测试集路径。然后可以通过运行以下命令开始测试：
```Shell
python tools/eval.py -c=configs/det/dbnet/db_r50_icdar15.yaml
```

MindOCR还支持在命令行中指定参数，可以运行以下的命令：
```Shell
python tools/eval.py -c=configs/det/dbnet/db_r50_icdar15.yaml \
            --opt eval.ckpt_load_path="/path/to/local_ckpt.ckpt" \
                  eval.dataset_root="/path/to/val_set/root" \
                  eval.data_dir="val_set/dir"\
                  eval.label_file="val_set/label"
```

### 3.3 推理

MindOCR推理支持Ascend310/Ascend310P设备，支持[MindSpore Lite](https://www.mindspore.cn/lite)和 [ACL](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/aclcppdevg/aclcppdevg_000004.html) 推理后端。推理教程给出了如何使用MindOCR进行推理的详细步骤，主要包括三个步骤：环境准备、模型转换和推理。

#### 3.3.1 环境准备

请参考[环境安装](../inference/environment.md)获取更多信息，并根据模型注意选择ACL/Lite环境。

#### 3.3.2 模型转换

在运行推理之前，用户需要从训练得到的checkpoint文件导出一个MindIR文件。MindSpore IR (MindIR)是基于图形表示的函数式IR。MindIR文件存储了推理所需的模型结构和权重参数。

根据训练好的dbnet checkpoint文件，用户可以使用以下命令导出MindIR：
```Shell
python tools/export.py --model_name_or_config dbnet_resnet50 --data_shape 736 1280 --local_ckpt_path /path/to/local_ckpt.ckpt
# 或者
python tools/export.py --model_name_or_config configs/det/dbnet/db_r50_icdar15.yaml --data_shape 736 1280 --local_ckpt_path /path/to/local_ckpt.ckpt
```

`data_shape`是MindIR文件的模型输入图片的高度和宽度。当用户使用其他的模型时，`data_shape`可能会改变。

请参考[转换教程](../inference/convert_tutorial.md)获取更多关于模型转换的细节。

#### 3.3.3 推理 (Python)

经过模型转换后， 用户能得到`output.mindir`文件。用户可以进入到`deploy/py_infer`目录，并使用以下命令进行推理：

```Shell
python infer.py \
    --input_images_dir=/your_path_to/test_images \
    --device=Ascend \
    --device_id=0 \
    --det_model_path=your_path_to/output.mindir \
    --det_model_name_or_config=../../configs/det/dbnet/db_r50_icdar15.yaml \
    --backend=lite \
    --res_save_dir=results_dir
```

请参考[推理教程](../inference/inference_tutorial.md)的`4.1 命令示例`章节获取更多例子。
