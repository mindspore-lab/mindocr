English | [中文](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/robustscanner/README_CN.md)

# RobustScanner
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [RobustScanner: Dynamically Enhancing Positional Clues for Robust Text Recognition](https://arxiv.org/pdf/2007.07542.pdf)

## 1. Introduction
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->

RobustScanner is an encoder-decoder text recognition algorithm with attention mechanism. The authors of this paper conducted research on the mainstream encoder-decoder recognition frameworks and found that during the decoding process, text not only relies on contextual information but also utilizes positional information. However, most methods rely too much on context information during the decoding process, leading to serious attention shifting problems and thus result in poor performance for text recognition with weak context information or contextless information.

To address this issue, the authors proposed a novel position-enhancement branch and dynamically fused its output with the output of the decoder attention module. The position-enhancement branch includes a position-aware module, a position embedding layer, and an attention module. The position-aware module enhances the output feature map of the encoder to encode rich position information. The position embedding layer takes the current decoding step as input and encodes it into a vector.

Overall, the RobustScanner model consists of an encoder and a decoder. The encoder uses ResNet-31 to extract features from the image. The decoder includes a hybrid branch and a position-enhancement branch, and the outputs of the two branches are dynamically fused and used to predict the final result. Based on the special design for position information, RobustScanner achieved the state-of-the-art results on both regular and irregular text recognition benchmarks and showed robustness in both contextual and non-contextual application scenarios.

<p align="center">
  <img src="https://github.com/tonytonglt/mindocr/assets/54050944/7c11121b-1962-4d29-93b6-5c9533992b8f" width=640 />
</p>
<p align="center">
  <em> Figure 1. Overall RobustScanner architecture [<a href="#references">1</a>] </em>
</p>

## 2. Results
<!--- Guideline:
Table Format:
- Model: model name in lower case with _ seperator.
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Top-1 and Top-5: Keep 2 digits after the decimal point.
- Params (M): # of model parameters in millions (10^6). Keep 2 digits after the decimal point
- Recipe: Training recipe/configuration linked to a yaml config file. Use absolute url path.
- Download: url of the pretrained model weights. Use absolute url path.
-->

### Accuracy

According to our experiments, the evaluation results on public benchmark datasets (IC03, IC13, IC15, IIIT, SVT, SVTP, CUTE) is as follow:

<div align="center">

|    **Model**     |    **Context**    | **Backbone**  | **Avg Accuracy** |       **Train T.**        | **FPS** | **ms/step** |                                                     **Recipe**                                                     |                                                                                                             **Download**                                                                                                              |
|:-------------:|:--------------:|:---------:|:---------:|:---------------------:|:-------:|:-----------:|:----------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| RobustScanner | D910x4-MS2.0-G | ResNet-31 |  87.86%   |     22560 s/epoch     |   310   |     825     | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/robustscanner/robustscanner_resnet31.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/robustscanner/robustscanner_resnet31-f27eab37.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/robustscanner/robustscanner_resnet31-f27eab37-158bde10.mindir) |
</div>

Note: In addition to using the MJSynth (partial) and SynthText (partial) text recognition datasets, RobustScanner is also trained with the SynthAdd dataset and some real datasets. The specific details of the data can be found in the paper or [here](#312-Dataset-Download).

<details open markdown>
  <div align="center">
  <summary>Detailed accuracy results for each benchmark dataset</summary>

| **Model** | **Backbone** | **IIIT5k** | **SVT** | **IC13** | **IC15** | **SVTP** | **CUTE80** | **Average** |
| :------: | :------: |:----------:|:-------:|:--------:|:--------:|:--------:|:----------:|:-----------:|
| RobustScanner  | ResNet-31 |   95.50%   | 92.12%  |  94.29%  |  73.33%  |  82.33%  |   89.58%   |   87.86%    |
  </div>
</details>

**Notes:**
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G-graph mode or F-pynative mode with ms function. For example, D910x4-MS1.10-G is for training on 4 pieces of Ascend 910 NPU using graph mode based on Minspore version 1.10.
- To reproduce the result on other contexts, please ensure the global batch size is the same.
- The model uses an English character dictionary, en_dict90.txt, consisting of 90 characters including digits, common symbols, and upper and lower case English letters. More explanation on dictionary, please refer to [4. Character Dictionary](#4-character-dictionary).
- The models are trained from scratch without any pre-training. For more dataset details of training and evaluation, please refer to [Dataset Download & Dataset Usage](#312-dataset-download) section.
- The input Shapes of MindIR of RobustScanner is (1, 3, 48, 160) and it is for Ascend only.

## 3. Quick Start
### 3.1 Preparation

#### 3.1.1 Installation
Please refer to the [installation instruction](https://github.com/mindspore-lab/mindocr#installation) in MindOCR.

#### 3.1.2 Dataset Download
The dataset used for training and validation in this work, was referenced from the datasets used by mmocr and PaddleOCR for reproducing the RobustScanner algorithms. We are very grateful to mmocr and PaddleOCR for improving the reproducibility efficiency of this repository.

The details of the dataset are as follows:
<div align="center">

| **Training set** |    **instance num**    | **repeat num** | **type** |
|:----------------:|:--------------:|:--------:|:--------:|
|    icdar2013     |      848       |    20    |   real   |
|    icdar2015     |      4468      |    20    |   real   |
|    coco_text     |     42142      |    20    |   real   |
|      IIIT5K      |      2000      |    20    |   real   |
|    SynthText     |    2400000     |    1     |  synth   |
|     SynthAdd     |    1216889     |    1     |    synth    |
|      Syn90k      |    2400000     |    1     |    synth    |

</div>
Note: SynthText and Syn90k datasets were both randomly sampled the complete SynthText dataset and complete Syn90k dataset,  consisting of 2.4 million samples each.

The training and evaluation datasets in LMDB format shown in the table above can be downloaded from here: [training set](https://aistudio.baidu.com/aistudio/datasetdetail/138433), [evaluation set](https://aistudio.baidu.com/aistudio/datasetdetail/138872)

The downloaded file contains several compressed files, including:
- Training set
  - `training_lmdb_real.zip`: contains four real datasets,IIIT5K, icdar2013, icdar2015 and coco_text, which are repeated 20 times during training;
  - `SynthAdd.zip`: contains the complete SynthAdd dataset;
  - `synth90K_shuffle.zip`: contains 2.4 million randomly sampled samples from the Synth90k dataset;
  - `SynthText800K_shuffle_xxx_xxx.zip`: includes 5 zip files, 1_200, and a total of 2.4 million randomly sampled samples from the SynthText dataset.
- Evaluation set
  - `testing_lmdb.zip`: contains six datasets used for evaluating the model, including CUTE80, icdar2013, icdar2015, IIIT5k, SVT, and SVTP.


#### 3.1.3 Dataset Usage

The data folder should be unzipped following the directory structure below:

``` text
data/
├── training
│   ├── real_data
│   │   ├── repeat1
│   │   │   ├── COCO_Text
│   │   │   │   ├── data.mdb
│   │   │   │   └── lock.mdb
│   │   │   ├── ICDAR2013
│   │   │   │   ├── data.mdb
│   │   │   │   └── lock.mdb
│   │   │   ├── ICDAR2015
│   │   │   │   ├── data.mdb
│   │   │   │   └── lock.mdb
│   │   │   └── IIIT5K
│   │   │       ├── data.mdb
│   │   │       └── lock.mdb
│   │   ├── repeat2
│   │   │   ├── COCO_Text
│   │   │   ├── ICDAR2013
│   │   │   ├── ICDAR2015
│   │   │   └── IIIT5K
│   │   │
│   │   ├── ...
│   │   │
│   │   └── repeat20
│   │       ├── COCO_Text
│   │       ├── ICDAR2013
│   │       ├── ICDAR2015
│   │       └── IIIT5K
│   └── synth_data
│       ├── synth90K_shuffle
│       │   ├── data.mdb
│       │   └── lock.mdb
│       ├── SynthAdd
│       │   ├── data.mdb
│       │   └── lock.mdb
│       ├── SynthText800K_shuffle_1_40
│       │   ├── data.mdb
│       │   └── lock.mdb
│       ├── SynthText800K_shuffle_41_80
│       │   ├── data.mdb
│       │   └── lock.mdb
│       └── ...
└── evaluation
    ├── CUTE80
    │   ├── data.mdb
    │   └── lock.mdb
    ├── IC13_1015
    │   ├── data.mdb
    │   └── lock.mdb
    ├── IC15_2077
    │   ├── data.mdb
    │   └── lock.mdb
    ├── IIIT5k_3000
    │   ├── data.mdb
    │   └── lock.mdb
    ├── SVT
    │   ├── data.mdb
    │   └── lock.mdb
    └── SVTP
        ├── data.mdb
        └── lock.mdb

```
Here, we use the datasets under the `training/` folder for training and the datasets under the `evaluation/` folder for model evaluation. For convenience of storage and usage, all data is in the lmdb format.


**Data configuration for model training**

To reproduce the training of model, it is recommended that you modify the configuration yaml as follows:

```yaml
...
train:
  ...
  dataset:
    type: LMDBDataset
    dataset_root: path/to/data/                           # Root dir of training dataset
    data_dir: training/                                   # Dir of training dataset, concatenated with `dataset_root` to be the complete dir of training dataset
...
eval:
  dataset:
    type: LMDBDataset
    dataset_root: path/to/data/                           # Root dir of validation dataset
    data_dir: evaluation/                                 # Dir of validation dataset, concatenated with `dataset_root` to be the complete dir of validation dataset
  ...
```

**Data configuration for model evaluation**

We use the dataset under `evaluation/` as the benchmark dataset. On **each individual dataset** (e.g. CUTE80, IC13_1015, etc.), we perform a full evaluation by setting the dataset's directory to the evaluation dataset. This way, we get a list of the corresponding accuracies for each dataset, and then the reported accuracies are the average of these values.

To reproduce the reported evaluation results, you can:
- Option 1: Repeat the evaluation step for all individual datasets: CUTE80, IC13_1015, IC15_2077, IIIT5k_3000, SVT, and SVTP. Then take the average score.

- Option 2: Put all the benchmark datasets folder under the same directory, e.g. `evaluation/`. And use the script `tools/benchmarking/multi_dataset_eval.py`.

1. Evaluate on one specific dataset

For example, you can evaluate the model on dataset `CUTE80` by modifying the config yaml as follows:

```yaml
...
train:
  # NO NEED TO CHANGE ANYTHING IN TRAIN SINCE IT IS NOT USED
...
eval:
  dataset:
    type: LMDBDataset
    dataset_root: path/to/data/                           # Root dir of evaluation dataset
    data_dir: evaluation/CUTE80/                          # Dir of evaluation dataset, concatenated with `dataset_root` to be the complete dir of evaluation dataset
  ...
```

By running `tools/eval.py` as noted in section [Model Evaluation](#33-model-evaluation) with the above config yaml, you can get the accuracy performance on dataset CUTE80.

2. Evaluate on multiple datasets under the same folder

Assume you have put all benckmark datasets under evaluation/ as shown below:

``` text
data/
├── evaluation
│   ├── CUTE80
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC03_860
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC03_867
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC13_1015
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── ...
```

then you can evaluate on each dataset by modifying the config yaml as follows, and execute the script `tools/benchmarking/multi_dataset_eval.py`.

```yaml
...
train:
  # NO NEED TO CHANGE ANYTHING IN TRAIN SINCE IT IS NOT USED
...
eval:
  dataset:
    type: LMDBDataset
    dataset_root: path/to/data/                           # Root dir of evaluation dataset
    data_dir: evaluation/                                 # Dir of evaluation dataset, concatenated with `dataset_root` to be the complete dir of evaluation dataset
  ...
```

#### 3.1.4 Check YAML Config Files
Apart from the dataset setting, please also check the following important args: `system.distribute`, `system.val_while_train`, `common.batch_size`, `train.ckpt_save_dir`, `train.dataset.dataset_root`, `train.dataset.data_dir`,
`eval.ckpt_load_path`, `eval.dataset.dataset_root`, `eval.dataset.data_dir`, `eval.loader.batch_size`. Explanations of these important args:

```yaml
system:
  distribute: True                                                    # `True` for distributed training, `False` for standalone training
  amp_level: 'O0'
  seed: 42
  val_while_train: True                                               # Validate while training
  drop_overflow_update: False
common:
  ...
  batch_size: &batch_size 64                                          # Batch size for training
  ...
train:
  ckpt_save_dir: './tmp_rec'                                          # The training result (including checkpoints, per-epoch performance and curves) saving directory
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: path/to/data/                                       # Root dir of training dataset
    data_dir: training/                                               # Dir of training dataset, concatenated with `dataset_root` to be the complete dir of training dataset
...
eval:
  ckpt_load_path: './tmp_rec/best.ckpt'                               # checkpoint file path
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: path/to/data/                                       # Root dir of validation/evaluation dataset
    data_dir: evaluation/                                             # Dir of validation/evaluation dataset, concatenated with `dataset_root` to be the complete dir of validation/evaluation dataset
  ...
  loader:
      shuffle: False
      batch_size: 64                                                  # Batch size for validation/evaluation
...
```
**Notes:**
- As the global batch size  (batch_size x num_devices) is important for reproducing the result, please adjust `batch_size` accordingly to keep the global batch size unchanged for a different number of GPUs/NPUs, or adjust the learning rate linearly to a new global batch size.


### 3.2 Model Training
<!--- Guideline: Avoid using shell script in the command line. Python script preferred. -->

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please modify the configuration parameter `distribute` as True and run

```shell
# distributed training on multiple GPU/Ascend devices
mpirun --allow-run-as-root -n 4 python tools/train.py --config configs/rec/robustscanner/robustscanner_resnet31.yaml
```


* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please modify the configuration parameter`distribute` as False and run:

```shell
# standalone training on a CPU/GPU/Ascend device
python tools/train.py --config configs/rec/robustscanner/robustscanner_resnet31.yaml
```

The training result (including checkpoints, per-epoch performance and curves) will be saved in the directory parsed by the arg `ckpt_save_dir`. The default directory is `./tmp_rec`.

### 3.3 Model Evaluation

To evaluate the accuracy of the trained model, you can use `eval.py`. Please set the checkpoint path to the arg `ckpt_load_path` in the `eval` section of yaml config file, set `distribute` to be False, and then run:

```shell
python tools/eval.py --config configs/rec/robustscanner/robustscanner_resnet31.yaml
```

## 4. Character Dictionary

### Default Setting

To transform the groud-truth text into label ids, we have to provide the character dictionary where keys are characters and values ​​are IDs. By default, the dictionary is **"0123456789abcdefghijklmnopqrstuvwxyz"**, which means id=0 will correspond to the character "0". In this case, the dictionary only considers numbers and lowercase English characters, excluding spaces.

### Built-in Dictionaries

There are some built-in dictionaries, which are placed in `mindocr/utils/dict/`, and you can choose the appropriate dictionary to use.

- `en_dict90.txt` is an English dictionary containing 90 characters, including numbers, common symbols, and uppercase and lowercase English letters.
- `en_dict.txt` is an English dictionary containing 94 characters, including numbers, common symbols, and uppercase and lowercase English letters.
- `ch_dict.txt` is a Chinese dictionary containing 6623 characters, including commonly used simplified and traditional Chinese, numbers, common symbols, uppercase and lowercase English letters.


### Customized Dictionary

You can also customize a dictionary file (***.txt) and place it under `mindocr/utils/dict/`, the format of the dictionary file should be a .txt file with one character per line.


To use a specific dictionary, change the parameter `model->head->out_channels` to the corresponding number, which is the number of characters in the dictionary + 3, change `model->head->start_idx`to the number of characters in the dictionary + 1, change `model->head->padding_idx`to the number of characters in the dictionary + 2, and lastly change `loss->ignore_index` to the number of characters in the dictionary + 2

```yaml
...
model:
  type: rec
  transform: null
  backbone:
    name: rec_resnet31
    pretrained: False
  head:
    name: RobustScannerHead
    out_channels: 93                 # modify to the number of characters in the dictionary + 3
    enc_outchannles: 128
    hybrid_dec_rnn_layers: 2
    hybrid_dec_dropout: 0.
    position_dec_rnn_layers: 2
    start_idx: 91                    # modify to the number of characters in the dictionary + 1
    mask: True
    padding_idx: 92                  # modify to the number of characters in the dictionary + 2
    encode_value: False
    max_text_len: *max_text_len
...

loss:
  name: SARLoss
  ignore_index: 92                   # modify to the number of characters in the dictionary + 2

...
```

**Notes:**
- You can include the space character by setting the parameter `use_space_char` in configuration yaml to True.
- Remember to check the value of `dataset->transform_pipeline->SARLabelEncode->lower` in the configuration yaml. Set it to False if you prefer case-sensitive encoding.

## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->
[1] Xiaoyu Yue, Zhanghui Kuang, Chenhao Lin, Hongbin Sun, Wayne Zhang. RobustScanner: Dynamically Enhancing Positional Clues for Robust Text Recognition. arXiv:2007.07542, ECCV'2020
