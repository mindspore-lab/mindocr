[English](https://github.com/mindspore-lab/mindocr/blob/main/configs/table/README.md) | 中文

# TableMaster
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [TableMaster: PINGAN-VCGROUP’S SOLUTION FOR ICDAR 2021 COMPETITION ON SCIENTIFIC LITERATURE PARSING TASK B: TABLE RECOGNITION TO HTML](https://arxiv.org/pdf/2105.01848.pdf)

## 1. 模型描述
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->

TableMaster是一种用于表格识别的模型，其独特之处在于能够同时训练得到单元格内的文字块位置和表格结构。传统的表格识别方法通常先回归出单元格的坐标，然后根据坐标获取表格的行列信息。然而，对于无表格线的情况，直接获取单元格位置或表格线信息变得困难。TableMaster提出了一种新的解决思路，采用超文本标记语言(HTML)作为预测目标，在训练过程中同时学习单元格内的文字块位置和表格结构。

网络结构包括编码和解码两个部分。编码阶段通过残差连接结构和多头通道注意力模块将图片转换为序列特征，便于后续的解码过程。编码阶段输出的序列特征经过位置编码后，输入到解码阶段。TableMaster的解码部分与Master相似，但增加了一个额外的分支。在一个Transformer层之后，解码部分分成两个分支，每个分支再经过两个Transformer层。这两个分支分别处理两个学习任务：单元格文字框的回归和表格结构序列的预测。

通过这种方式，TableMaster能够同时学习和预测单元格的位置和表格的结构，提高了表格识别的准确性和效果。它的独特设计和训练策略使得在无表格线的场景下也能准确获取表格信息，具有广泛的应用场景。

<p align="center">
  <img src="https://github.com/tonytonglt/mindocr-fork/assets/54050944/556ad4a5-d892-44c4-9d57-c22f6f5510fc" width=480 />
</p>
<p align="center">
  <em> 图1. TableMaster整体架构图 [<a href="#参考文献">1</a>] </em>
</p>

## 2. 实验结果

### PubTabNet
<div align="center">

| **模型**      | **环境配置**        | **骨干网络**    | **Accuracy** | **训练时间**     | **每步耗时**    | **FPS**   | **配置文件**                  | 模型权重下载                                                                                                     |
|-------------|-----------------|-------------|--------------|--------------|-------------|-----------|---------------------------|------------------------------------------------------------------------------------------------------------|
| TableMaster | D910*x8-MS2.4-F | TableResNetExtra   | 77.47%       | 4218 s/epoch | 675 ms/step | 120 img/s | [yaml](table_master.yaml) | [ckpt](https://download-mindspore.osinfra.cn/toolkits/mindocr/tablemaster/table_master-78bf35bb.ckpt) |
| TableMaster | D910*x8-MS2.3-G | TableResNetExtra   | 77.49%       | 1675 s/epoch | 268 ms/step | 296 img/s | [yaml](table_master.yaml) | [ckpt](https://download-mindspore.osinfra.cn/toolkits/mindocr/tablemaster/table_master-78bf35bb.ckpt) |

</div>

#### 注释：
- 环境配置：训练的环境配置表示为 {处理器}x{处理器数量}-{MS模式}，其中 Mindspore 模式可以是 G-graph 模式或 F-pynative 模式。
- TableMaster的训练时长受数据处理部分和不同运行环境的影响非常大。
- 链接中MindIR导出时的输入Shape为`(1,3,480,480)` 。


## 3. 快速上手

### 3.1 安装

请参考MindOCR套件的[安装指南](https://github.com/mindspore-lab/mindocr#installation) 。

### 3.2 数据准备

请从[该网址](https://github.com/ibm-aur-nlp/PubTabNet)下载PubTabNet数据集，对zip文件进行解压，并根据标注文件`PubTabNet_2.0.0.jsonl`中的`split`字段将数据划分为训练集和验证集。

完成数据准备工作后，数据的目录结构应该如下所示：


``` text
PubTabNet
├── train
│   ├── PMC1064074_007_00.png
│   ├── PMC1064076_003_00.png
│   ├── PMC1064076_004_00.png
│   └── ....png
│  
├── val
│   ├── PMC1064865_002_00.png
│   ├── PMC1079806_002_00.png
│   ├── PMC1079811_004_00.png
│   └── ....png
│
├── PubTabNet_2.0.0_train.jsonl
│
└── PubTabNet_2.0.0_val.jsonl
```

### 3.3 配置说明

在配置文件`configs/table/table_master.yaml`中更新如下文件路径。其中`dataset_root`为训练集图片文件夹目录，`label_file_list`为训练集标签文件路径列表，可包含多个标签文件路径。

```yaml
...
train:
  ckpt_save_dir: './tmp_table'
  dataset_sink_mode: False
  dataset:
    type: PubTabDataset
    dataset_root: dir/to/train                             # <--- 更新
    label_file_list: [dir/to/PubTabNet_2.0.0_train.jsonl]  # <--- 更新
    sample_ratio_list: [ 1.0 ]
...
eval:
  dataset_sink_mode: False
  dataset:
    type: PubTabDataset
    dataset_root: dir/to/val                               # <--- 更新
    label_file_list: [dir/to/PubTabNet_2.0.0_val.jsonl]    # <--- 更新
    sample_ratio_list: [ 1.0 ]
...
```

> 【可选】可以根据CPU核的数量设置`num_workers`参数的值。



TableMaster由2个部分组成：`backbone`和`head`。如下所示:

```yaml
model:
  type: table
  transform: null
  backbone:
    name: table_resnet_extra
    gcb_config:
      ratio: 0.0625
      headers: 1
      att_scale: False
      fusion_type: channel_add
      layers: [ False, True, True, True ]
    layers: [ 1,2,5,3 ]
  head:
    name: TableMasterHead
    out_channels: 43
    hidden_size: 512
    headers: 8
    dropout: 0.
    d_ff: 2024
    max_text_length: *max_text_len
    loc_reg_num: &loc_reg_num 4
```

### 3.4 训练

* 单卡训练

请确保yaml文件中的`distribute`参数为False。

``` shell
# train tablemaster on pubtabnet dataset
python tools/train.py --config configs/table/table_master.yaml
```

* 分布式训练

请确保yaml文件中的`distribute`参数为True。

```shell
# n is the number of GPUs/NPUs
mpirun --allow-run-as-root -n 8 python tools/train.py --config configs/table/table_master.yaml
```

训练结果（包括checkpoint、每个epoch的性能和曲线图）将被保存在yaml配置文件的`ckpt_save_dir`参数配置的路径下，默认为`./tmp_table`。

### 3.5 评估

评估环节，在yml配置文件中将`ckpt_load_path`参数配置为checkpoint文件的路径，设置`distribute`为False，然后运行：

``` shell
python tools/eval.py --config configs/table/table_master.yaml
```
