English | [中文](https://github.com/mindspore-lab/mindocr/blob/main/configs/table/README_CN.md)

# TableMaster
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [TableMaster: PINGAN-VCGROUP’S SOLUTION FOR ICDAR 2021 COMPETITION ON SCIENTIFIC LITERATURE PARSING TASK B: TABLE RECOGNITION TO HTML](https://arxiv.org/pdf/2105.01848.pdf)

## Introduction

<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->

TableMaster is a model used for table recognition, and its uniqueness lies in its ability trained for simultaneously recognizing both the position of text blocks within cells and the structure of the table. Traditional methods for table recognition usually involve regressing the coordinates of cells and then obtaining the row and column information based on those coordinates. However, it becomes challenging to directly obtain cell positions or table line information in cases where there are no table lines. TableMaster proposes a new solution by simultaneously learning the position of text blocks within cells and the structure of the table during the training process, using a representation based on Hypertext Markup Language (HTML).

The network architecture consists of two parts: encoding and decoding. In the encoding phase, the image is transformed into sequential features using residual connections and multi-head channel attention modules, facilitating the subsequent decoding process. The sequential features outputted from the encoding phase are then inputted into the decoding phase after position encoding. The decoding part of TableMaster is similar to Master but with an additional branch. After a Transformer layer, the decoding part splits into two branches, each going through two more Transformer layers. These two branches handle two learning tasks respectively: regression of cell text boxes and prediction of the table structure sequence.

Through this approach, TableMaster is able to simultaneously learn and predict the position of cells and the structure of the table, thereby improving the accuracy and effectiveness of table recognition. Its unique design and training strategy enable accurate retrieval of table information even in scenarios without table lines, making it applicable in a wide range of contexts.

<p align="center">
  <img src="https://github.com/tonytonglt/mindocr-fork/assets/54050944/556ad4a5-d892-44c4-9d57-c22f6f5510fc" width=480 />
</p>
<p align="center">
  <em> Figure 1. Overall TableMaster architecture [<a href="#references">1</a>] </em>
</p>

## Requirements

| mindspore  | ascend driver  |    firmware    | cann toolkit/kernel |
|:----------:|:--------------:|:--------------:|:-------------------:|
|   2.5.0    |    24.1.0      |   7.5.0.3.220  |     8.0.0.beta1     |

## Quick Start

### Installation

Please refer to the [installation instruction](https://github.com/mindspore-lab/mindocr#installation) in MindOCR.

### Dataset preparation

#### PubTabNet dataset

Please download [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet) dataset, unzip the zip files, and split the annotation file into training set and validation set according to the `split` key in the `PubTabNet_2.0.0.jsonl` file.


The prepared dataset file struture should be:


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


### Configuration Description

Update `configs/table/table_master.yaml`configuration file with data paths,
specifically`dataset_root` is the directory of training set images folder, `label_file_list` is a list of training set annotation file path, and it may include multiple annotation file paths.

```yaml
...
train:
  ckpt_save_dir: './tmp_table'
  dataset_sink_mode: False
  dataset:
    type: PubTabDataset
    dataset_root: dir/to/train                             # <--- Update
    label_file_list: [dir/to/PubTabNet_2.0.0_train.jsonl]  # <--- Update
    sample_ratio_list: [ 1.0 ]
...
eval:
  dataset_sink_mode: False
  dataset:
    type: PubTabDataset
    dataset_root: dir/to/val                               # <--- Update
    label_file_list: [dir/to/PubTabNet_2.0.0_val.jsonl]    # <--- Update
    sample_ratio_list: [ 1.0 ]
...
```

> Optionally, change `num_workers` according to the number of CPU cores.



TableMaster consists of 2 parts: `backbone` and `head`. Specifically:

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

### Training

* Standalone training

Please set `distribute` to False in yaml config file .

``` shell
# train tablemaster on pubtabnet dataset
python tools/train.py --config configs/table/table_master.yaml
```

* Distributed training

Please set `distribute` in yaml config file to be True.

```shell
# worker_num is the total number of Worker processes participating in the distributed task.
# local_worker_num is the number of Worker processes pulled up on the current node.
# The number of processes is equal to the number of NPUs used for training. In the case of single-machine multi-card worker_num and local_worker_num must be the same.
msrun --worker_num=8 --local_worker_num=8 python tools/train.py --config configs/table/table_master.yaml

# Based on verification,binding cores usually results in performance acceleration.Please configure the parameters and run.
msrun --bind_core=True --worker_num=8 --local_worker_num=8 python tools/train.py --config configs/table/table_master.yaml
```
**Note:** For more information about msrun configuration, please refer to [here](https://www.mindspore.cn/docs/en/r2.5.0/model_train/parallel/msrun_launcher.html).


The training result (including checkpoints, per-epoch performance and curves) will be saved in the directory parsed by the arg `ckpt_save_dir` in yaml config file. The default directory is `./tmp_table`.

### Evaluation

To evaluate the accuracy of the trained model, you can use `eval.py`. Please set the checkpoint path to the arg `ckpt_load_path` in the `eval` section of yaml config file, set `distribute` to be False, and then run:

``` shell
python tools/eval.py --config configs/table/table_master.yaml
```

## Performance

### PubTabNet

Experiments are tested on ascend 910* with mindspore 2.5.0 graph mode
<div align="center">

| **model name** | **cards** | **batch size** | **ms/step** | **img/s** | **accuracy** | **config**  | **weight**                                                                            |
|----------------|-----------|----------------|-------------|-----------|--------------|-----------------------------------------------------|------------------------------------------------|
| TableMaster         | 8         | 10             | 268         | 296       | 77.49%       | [yaml](table_master.yaml) | [ckpt](https://download-mindspore.osinfra.cn/toolkits/mindocr/tablemaster/table_master-78bf35bb.ckpt) |
</div>

#### Notes：
- The training time of TableMaster is highly affected by data processing and varies on different machines.
- The input_shape for exported MindIR in the link is `(1,3,480,480)`.
