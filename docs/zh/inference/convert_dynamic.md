## 推理 - 动态Shape分档

### 1. 简介

在某些推理场景，如检测出目标后再进行目标识别，由于目标个数和目标大小不固定，导致目标识别网络输入BatchSize和ImageSize不固定。如果每次推理都按照最大的BatchSize或最大ImageSize进行计算，会造成计算资源浪费。

所以，在模型转换时可以设置一些候选值，推理的时候Resize到最匹配的候选值，从而提高性能。用户可以凭经验手动选择这些候选值，也可以从数据集中统计而来。

本工具集成了数据集统计的功能，可以统计出合适的`batch size`、`height`和`width`组合作为候选值，并封装了模型转换工具，从而实现了自动化分档功能。

### 2. 运行环境

请参考[环境安装](environment.md)，安装MindSpore Lite环境。

### 3. 模型准备

当前支持输入MindIR/ONNX模型文件，自动分档并转换为MIndIR模型文件。

请确保输入模型为动态Shape版的。例如，文本检测模型如果需要对H和W分档，要确保至少H和W轴是动态的，Shape可以为`(1,3,-1,-1)`和`(-1,3,-1,-1)`等。

### 4. 数据集准备

支持两种类型的数据：

1. 图像文件夹

   - 该工具会读取文件夹下的所有图像，记录`height`和`width`，统计出合适的候选值

   - 适合文本检测和文本识别模型

2. 文本检测的标注文件

   - 可参考[converter](../datasets/converters.md)，它是参数`task`为`det`时输出的标注文件

   - 该工具会读取每张图像下标注的文本框坐标，记录`height`和`width`，以及框的数量作为`batch size`，统计出合适的候选值

   - 适合文本识别模型

#### 5. 用法

`cd deploy/models_utils/auto_scaling`

##### 5.1 命令示例

- 对batch size进行分档

```shell
python converter.py \
    --model_path=/path/to/model.mindir \
    --dataset_path=/path/to/det_gt.txt
    --input_shape=-1,3,48,192 \
    --output_path=output
```

输出结果为单个OM模型：`model_dynamic_bs.mindir`

- 对height和width进行分档

```shell
python converter.py \
    --model_path=/path/to/model.onnx \
    --dataset_path=/path/to/images \
    --input_shape=1,3,-1,-1 \
    --output_path=output
```

输出结果为单个OM模型：`model_dynamic_hw.mindir`

- 对batch size、height和width进行分档

```shell
python converter.py \
    --model_path=/path/to/model.onnx \
    --dataset_path=/path/to/images \
    --input_shape=-1,3,-1,-1 \
    --output_path=output
```

输出结果为多个MindIR模型，组合了多个不同Batch Size，每个模型使用相同的动态Image Size：`model_dynamic_bs1_hw.mindir`, `model_dynamic_bs4_hw.mindir`, ......

- 不做分档

```shell
python converter.py \
    --model_path=/path/to/model.onnx \
    --input_shape=4,3,48,192 \
    --output_path=output
```

输出结果为单个MindIR模型：`model_static.mindir`

##### 5.2 详细参数

| 名称        | 默认值      | 必需 | 含义                                    |
| ----------- | ----------- | ---- | --------------------------------------- |
| model_path  | 无          | 是   | 模型文件路径                            |
| input_shape | 无          | 是   | 模型输入shape，NCHW格式                 |
| data_path   | 无          | 否   | 数据集或标注文件的路径                  |
| input_name  | x           | 否   | 模型的输入名                            |
| backend     | lite         | 否   | 转换工具, lite或者acl                  |
| output_path | ./output    | 否   | 输出模型保存文件夹                      |
| soc_version | Ascend310P3 | 否   | Ascend的soc型号，Ascend310P3或Ascend310 |

##### 5.3 配置文件

除了上述命令行参数外，在[auto_scaling.yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/models_utils/auto_scaling/configs/auto_scaling.yaml)中还有一些参数，用以描述数据集的统计方式，如有需要可自行修改：

- limit_side_len

   原始输入数据的`height`和`width`大小限制，超出范围按照比例进行压缩，可以调整数据的离散程度。

- strategy

   数据统计算法策略，支持`mean_std`和`max_min`两种算法，默认：`mean_std`。

   - mean_std

    ```
    mean_std = [mean - n_std * sigma，mean + n_std * sigma]
    ```
   - max_min
    ```
    max_min = [min - (max - min) * expand_ratio / 2，max + (max - min) * expand_ratio / 2]
    ```

- width_range/height_range

  对离散统计之后的width/height大小限制，超出将被过滤。

- interval

   间隔大小，如某些网络可能要求输入尺寸必须是32的倍数。

- max_scaling_num

  分档数量的上限。

- batch_choices

  默认的batch size值，如果data_path传入的是图像文件夹，则无法统计出batch size信息，就会使用该默认值。

- default_scaling

  用户不传入data_path数据时，提供默认的`height`和`width`分档值。
