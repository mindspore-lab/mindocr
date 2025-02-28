English | [中文](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/svtr/README_CN.md)

# SVTR
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [SVTR: Scene Text Recognition with a Single Visual Model](https://arxiv.org/abs/2205.00159)

## Introduction
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->

Dominant scene text recognition models commonly contain two building blocks, a visual model for feature extraction and a sequence model for text transcription. This hybrid architecture, although accurate, is complex and less efficient. This paper proposes a Single Visual model for Scene Text recognition within the patch-wise image tokenization framework, which dispenses with the sequential modeling entirely. The method, termed SVTR, firstly decomposes an image text into small patches named character components. Afterward, hierarchical stages are recurrently carried out by component-level mixing, merging and/or combining. Global and local mixing blocks are devised to perceive the inter-character and intra-character patterns, leading to a multi-grained character component perception. Thus, characters are recognized by a simple linear prediction. Experimental results on both English and Chinese scene text recognition tasks demonstrate the effectiveness of SVTR. SVTR-L (Large) achieves highly competitive accuracy in English and outperforms existing methods by a large margin in Chinese, while running faster. In addition, SVTR-T (Tiny) is an effective and much smaller model, which shows appealing speed at inference. [<a href="#references">1</a>]

<!--- Guideline: If an architecture table/figure is available in the paper, put one here and cite for intuitive illustration. -->
<p align="center">
  <img src="https://github.com/zhtmike/mindocr/assets/8342575/27da30e5-f0af-4a11-afc8-a902785e44c1" width=450 />
</p>
<p align="center">
  <em> Figure 1. Architecture of SVTR [<a href="#references">1</a>] </em>
</p>



## Requirements

| mindspore  | ascend driver  |   firmware    | cann toolkit/kernel |
|:----------:|:--------------:|:-------------:|:-------------------:|
|   2.3.1    |    24.1.RC2    |  7.3.0.1.231  |   8.0.RC2.beta1     |


## Quick Start
### Preparation

#### Installation
Please refer to the [installation instruction](https://github.com/mindspore-lab/mindocr#installation) in MindOCR.

#### Dataset Preparation

##### MJSynth, validation and evaluation dataset
Part of the lmdb dataset for training and evaluation can be downloaded from [here](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0) (ref: [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here)). There're several zip files:
- `data_lmdb_release.zip` contains the datasets including training data, validation data and evaluation data.
    - `training/` contains two datasets: [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/) and [SynthText (ST)](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c). *Here we use **MJSynth only**.*
    - `validation/` is the union of the training sets of [IC13](http://rrc.cvc.uab.es/?ch=2), [IC15](http://rrc.cvc.uab.es/?ch=4), [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), and [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset).
    - `evaluation/` contains several benchmarking datasets, which are [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset), [IC03](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions), [IC13](http://rrc.cvc.uab.es/?ch=2), [IC15](http://rrc.cvc.uab.es/?ch=4), [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf), and [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html).
- `validation.zip`: same as the validation/ within data_lmdb_release.zip
- `evaluation.zip`: same as the evaluation/ within data_lmdb_release.zip

##### SynthText dataset

For `SynthText`, we do not use the given LMDB dataset in `data_lmdb_release.zip`, since it only contains part of the cropped images. Please download the raw dataset from [here](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c) and prepare the LMDB dataset using the following command

```bash
python tools/dataset_converters/convert.py \
    --dataset_name synthtext \
    --task rec_lmdb \
    --image_dir path_to_SynthText \
    --label_dir path_to_SynthText_gt.mat \
    --output_path ST_full
```
the `ST_full` contained the full cropped images of SynthText in LMDB data format. Please replace the `ST` folder with the `ST_full` folder.

#### Dataset Usage

Finally, the data structure should like this.

``` text
data_lmdb_release/
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
├── training
│   ├── MJ
│   │   ├── MJ_test
│   │   │   ├── data.mdb
│   │   │   └── lock.mdb
│   │   ├── MJ_train
│   │   │   ├── data.mdb
│   │   │   └── lock.mdb
│   │   └── MJ_valid
│   │       ├── data.mdb
│   │       └── lock.mdb
│   └── ST_full
│       ├── data.mdb
│       └── lock.mdb
└── validation
    ├── data.mdb
    └── lock.mdb
```

Here we used the datasets under `training/` folders for training, and the union dataset `validation/` for validation. After training, we used the datasets under `evaluation/` to evaluate model accuracy.

**Training:** (total 16,185,770 samples)
- [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/)
  - Train: 21.2 GB, 7224586 samples
  - Valid: 2.36 GB, 802731 samples
  - Test: 2.61 GB, 891924 samples
- [SynthText (ST)](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c)
  - Train: 17.0 GB, 7266529 samples

**Validation:**
- Valid: 138 MB, 6992 samples

**Evaluation:** (total 12,067 samples)
- [CUTE80](http://cs-chan.com/downloads_CUTE80_dataset.html): 8.8 MB, 288 samples
- [IC03_860](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions): 36 MB, 860 samples
- [IC03_867](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions): 4.9 MB, 867 samples
- [IC13_857](http://rrc.cvc.uab.es/?ch=2): 72 MB, 857 samples
- [IC13_1015](http://rrc.cvc.uab.es/?ch=2): 77 MB, 1015 samples
- [IC15_1811](http://rrc.cvc.uab.es/?ch=4): 21 MB, 1811 samples
- [IC15_2077](http://rrc.cvc.uab.es/?ch=4): 25 MB, 2077 samples
- [IIIT5k_3000](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html): 50 MB, 3000 samples
- [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset): 2.4 MB, 647 samples
- [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf): 1.8 MB, 645 samples


**Data configuration for model training**

To reproduce the training of model, it is recommended that you modify the configuration yaml as follows:

```yaml
...
train:
  ...
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # Root dir of training dataset
    data_dir: training/                                               # Dir of training dataset, concatenated with `dataset_root` to be the complete dir of training dataset
...
eval:
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # Root dir of validation dataset
    data_dir: validation/                                             # Dir of validation dataset, concatenated with `dataset_root` to be the complete dir of validation dataset
  ...
```

**Data configuration for model evaluation**

We use the dataset under `evaluation/` as the benchmark dataset. On **each individual dataset** (e.g. CUTE80, IC03_860, etc.), we perform a full evaluation by setting the dataset's directory to the evaluation dataset. This way, we get a list of the corresponding accuracies for each dataset, and then the reported accuracies are the average of these values.

To reproduce the reported evaluation results, you can:
- Option 1: Repeat the evaluation step for all individual datasets: CUTE80, IC03_860, IC03_867, IC13_857, IC131015, IC15_1811, IC15_2077, IIIT5k_3000, SVT, SVTP. Then take the average score.

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
    dataset_root: dir/to/data_lmdb_release/                           # Root dir of evaluation dataset
    data_dir: evaluation/CUTE80/                                      # Dir of evaluation dataset, concatenated with `dataset_root` to be the complete dir of evaluation dataset
  ...
```

By running `tools/eval.py` as noted in section [Model Evaluation](#model-evaluation) with the above config yaml, you can get the accuracy performance on dataset CUTE80.


2. Evaluate on multiple datasets under the same folder

Assume you have put all benckmark datasets under evaluation/ as shown below:

``` text
data_lmdb_release/
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
    dataset_root: dir/to/data_lmdb_release/                           # Root dir of evaluation dataset
    data_dir: evaluation/                                   # Dir of evaluation dataset, concatenated with `dataset_root` to be the complete dir of evaluation dataset
  ...
```

#### Check YAML Config Files
Apart from the dataset setting, please also check the following important args: `system.distribute`, `system.val_while_train`, `common.batch_size`, `train.ckpt_save_dir`, `train.dataset.dataset_root`, `train.dataset.data_dir`, `train.dataset.label_file`,
`eval.ckpt_load_path`, `eval.dataset.dataset_root`, `eval.dataset.data_dir`, `eval.dataset.label_file`, `eval.loader.batch_size`. Explanations of these important args:

```yaml
system:
  distribute: True                                                    # `True` for distributed training, `False` for standalone training
  amp_level: 'O2'
  amp_level_infer: "O2"
  seed: 42
  val_while_train: True                                               # Validate while training
  drop_overflow_update: False
common:
  ...
  batch_size: &batch_size 512                                         # Batch size for training
...
train:
  ckpt_save_dir: './tmp_rec'                                          # The training result (including checkpoints, per-epoch performance and curves) saving directory
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # Root dir of training dataset
    data_dir: training/                                               # Dir of training dataset, concatenated with `dataset_root` to be the complete dir of training dataset
...
eval:
  ckpt_load_path: './tmp_rec/best.ckpt'                               # checkpoint file path
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # Root dir of validation/evaluation dataset
    data_dir: validation/                                             # Dir of validation/evaluation dataset, concatenated with `dataset_root` to be the complete dir of validation/evaluation dataset
  ...
  loader:
      shuffle: False
      batch_size: 512                                                 # Batch size for validation/evaluation
...
```

**Notes:**
- As the global batch size  (batch_size x num_devices) is important for reproducing the result, please adjust `batch_size` accordingly to keep the global batch size unchanged for a different number of NPUs, or adjust the learning rate linearly to a new global batch size.


### Model Training
<!--- Guideline: Avoid using shell script in the command line. Python script preferred. -->

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please modify the configuration parameter `distribute` as True and run

```shell
# distributed training on multiple Ascend devices
# worker_num is the total number of Worker processes participating in the distributed task.
# local_worker_num is the number of Worker processes pulled up on the current node.
# The number of processes is equal to the number of NPUs used for training. In the case of single-machine multi-card worker_num and local_worker_num must be the same.
msrun --worker_num=8 --local_worker_num=8 python tools/train.py --config configs/rec/svtr/svtr_tiny_8p.yaml

# Based on verification,binding cores usually results in performance acceleration.Please configure the parameters and run.
msrun --bind_core=True --worker_num=8 --local_worker_num=8 python tools/train.py --config configs/rec/svtr/svtr_tiny_8p.yaml
```
**Note:** For more information about msrun configuration, please refer to [here](https://www.mindspore.cn/tutorials/experts/en/r2.3.1/parallel/msrun_launcher.html).



* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please modify the configuration parameter`distribute` as False and run:

```shell
# standalone training on a CPU/Ascend device
python tools/train.py --config configs/rec/svtr/svtr_tiny.yaml
```

The training result (including checkpoints, per-epoch performance and curves) will be saved in the directory parsed by the arg `ckpt_save_dir`. The default directory is `./tmp_rec`.

### Model Evaluation

To evaluate the accuracy of the trained model, you can use `eval.py`. Please set the checkpoint path to the arg `ckpt_load_path` in the `eval` section of yaml config file, set `distribute` to be False, and then run:

```shell
python tools/eval.py --config configs/rec/svtr/svtr_tiny.yaml
```

## Character Dictionary

### Default Setting

To transform the groud-truth text into label ids, we have to provide the character dictionary where keys are characters and values ​​are IDs. By default, the dictionary is **"0123456789abcdefghijklmnopqrstuvwxyz"**, which means id=0 will correspond to the charater "0". In this case, the dictionary only considers numbers and lowercase English characters, excluding spaces.

### Built-in Dictionaries

There are some built-in dictionaries, which are placed in `mindocr/utils/dict/`, and you can choose the appropriate dictionary to use.

- `en_dict.txt` is an English dictionary containing 94 characters, including numbers, common symbols, and uppercase and lowercase English letters.
- `ch_dict.txt` is a Chinese dictionary containing 6623 characters, including commonly used simplified and traditional Chinese, numbers, common symbols, uppercase and lowercase English letters.


### Customized Dictionary

You can also customize a dictionary file (***.txt) and place it under `mindocr/utils/dict/`, the format of the dictionary file should be a .txt file with one character per line.


To use a specific dictionary, set the parameter `character_dict_path` to the path of the dictionary, and change the parameter `num_classes` to the corresponding number, which is the number of characters in the dictionary + 1.


**Notes:**
- You can include the space character by setting the parameter `use_space_char` in configuration yaml to True.
- Remember to check the value of `dataset->transform_pipeline->RecAttnLabelEncode->lower` in the configuration yaml. Set it to False if you prefer case-sensitive encoding.


## Chinese Text Recognition Model Training

Currently, this model supports multilingual recognition and provides pre-trained models for different languages. Details are as follows:

### Chinese Dataset Preparation and Configuration

We use a public Chinese text benchmark dataset [Benchmarking-Chinese-Text-Recognition](https://github.com/FudanVI/benchmarking-chinese-text-recognition) for SVTR training and evaluation.

For detailed instruction of data preparation and yaml configuration, please refer to [ch_dataeset](../../../docs/en/datasets/chinese_text_recognition.md).


## Performance

### General Purpose Chinese Models

Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.

| **model name** | **cards** | **batch size** | **languages** | **jit level** | **graph compile** | **ms/step** | **img/s** | **scene** | **web** | **document** |                                                 **recipe**                                                 |                                                                                          **weight**                                                                                           |
| :------------: | :-------: | :------------: | :-----------: | :-----------: | :---------------: | :---------: | :-------: |:---------:|:-------:| :----------: | :--------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   SVTR-Tiny    |     4     |      256       |    Chinese    |      O2       |      235.1 s      |    37.75    |   1580    |  66.19%   | 69.66%  |    98.01%    | [svtr_tiny_ch.yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/svtr/svtr_tiny_ch.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/svtr/svtr_tiny_ch-2ee6ade4.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/svtr/svtr_tiny_ch-2ee6ade4-3e495768.mindir) |

### Specific Purpose Models

Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.

| **model name** | **backbone** |  **train dataset**   | **params(M)** | **cards** | **batch size** | **jit level** | **graph compile** | **ms/step** | **img/s** | **accuracy** |                                           **recipe**                                           |                                                                                                  **weight**                                                                                                   |
|:--------------:|:------------:|:--------------------:|:-------------:|:---------:|:--------------:| :-----------: |:-----------------:|:-----------:|:---------:|:------------:|:----------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  SVTR-Tiny-8P  |     Tiny     |        MJ+ST         |     60.24     |     8     |      512       |      O2       |     156.85 s      |   410.16    |  9986.34  |     90.29%   | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/svtr/svtr_tiny_8p.yaml)  | [ckpt](https://download-mindspore.osinfra.cn/toolkits/mindocr/svtr/svtr_tiny_8p-0afc75d6.ckpt) \|  [mindir](https://download-mindspore.osinfra.cn/toolkits/mindocr/svtr/svtr_tiny_8p-0afc75d6-255191ef.mindir) |

Detailed accuracy results for each benchmark dataset:

| **model name** | **backbone** | **cards** | **IC03_860** | **IC03_867** | **IC13_857** | **IC13_1015** | **IC15_1811** | **IC15_2077** | **IIIT5k_3000** | **SVT** | **SVTP** | **CUTE80** | **average** |
|:--------------:|:------------:|:---------:|:------------:|:------------:|:------------:|:-------------:|:-------------:|:-------------:|:---------------:|:-------:|:--------:|:----------:|:-----------:|
|  SVTR-Tiny-8P  |     Tiny     |     1     |    95.93%    |    95.62%    |    95.33%    |    93.89%     |    84.32%     |    80.55%     |     94.30%      | 90.42%  |  86.05%  |   86.46%   |   90.29%    |


### Notes
- To reproduce the result on other contexts, please ensure the global batch size is the same.
- The characters supported by model are lowercase English characters from a to z and numbers from 0 to 9. More explanation on dictionary, please refer to [Character Dictionary](#character-dictionary).
- The models are trained from scratch without any pre-training. For more dataset details of training and evaluation, please refer to [Dataset Preparation](#dataset-preparation) section.
- The input Shapes of MindIR of RARE is (1, 3, 64, 256).


## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Yongkun Du, Zhineng Chen, Caiyan Jia, Xiaoting Yin, Tianlun Zheng, Chenxia Li, Yuning Du, Yu-Gang Jiang. SVTR: Scene Text Recognition with a Single Visual Model. arXiv preprint arXiv:2205.00159, 2022.
