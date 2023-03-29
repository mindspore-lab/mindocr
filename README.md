
# MindOCR

<!--
[![license](https://img.shields.io/github/license/mindspore-lab/mindocr.svg)](https://github.com/mindspore-lab/mindocr/blob/main/LICENSE)
[![open issues](https://img.shields.io/github/issues/mindspore-lab/mindocr)](https://github.com/mindspore-lab/mindocr/issues)
[![PRs](https://img.shields.io/badge/PRs-welcome-pink.svg)](https://github.com/mindspore-lab/mindocr/pulls)
 -->
English | [中文](README_CN.md)

[Introduction](#introduction) |
[Installation](#installation) |
[Quick Start](#quick-start) |
[Model List](#supported-models-and-performance) |
[Notes](#notes)


## Introduction
MindOCR is an open-source toolbox for OCR development and application based on [MindSpore](https://www.mindspore.cn/en). It helps users to train and apply the best text detection and recognition models, such as DBNet/DBNet++ and CRNN/SVTR, to fulfuill image-text understanding need.


<details open>
<summary> Major Features </summary>

- **Modulation design**: We decouple the ocr task into serveral configurable modules. Users can setup the training and evaluation pipeline easily for customized data and models with a few line of modification.
- **High-performance**: MindOCR provides pretrained weights and the used training recipes that reach competitive performance on OCR tasks.
- **Low-cost-to-apply**: We provide easy-to-use tools to run text detection and recogintion on real-world data. (coming soon)
</details>


## Installation

### Dependency

To install the dependency, please run
```shell
pip install -r requirements.txt
```

Additionally, please install MindSpore(>=1.9) following the official [instructions](https://www.mindspore.cn/install) for the best fit of your machine. 

For distributed training, please install [openmpi 4.0.3](https://www.open-mpi.org/software/ompi/v4.0/).


### Install with PyPI

Coming soon

### Install from Source

The latest version of MindOCR can be installed as follows:
```shell
pip install git+https://github.com/mindspore-lab/mindocr.git
```

> Notes: MindOCR is only tested on MindSpore>=1.8.1, Linux on GPU/Ascend devices currently.

## Quick Start

### Text Detection Model Training

We will use **DBNet** model and **ICDAR2015** dataset for illustration, although other models and datasets are also supported. <!--ICDAR15 is a commonly-used model and a benchmark for scene text recognition.-->

#### 1. Data Preparation

Please download the ICDAR2015 dataset from this [website](https://rrc.cvc.uab.es/?ch=4&com=downloads), then format the dataset annotation refer to [dataset_convert](tools/dataset_converters/README.md).

After preparation, the data structure should be like 

``` text
.
├── test
│   ├── images
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   └── ...
│   └── det_gt.txt
└── train
    ├── images
    │   ├── img_1.jpg
    │   ├── img_2.jpg
    │   └── ....jpg
    └── det_gt.txt
```

#### 2. Configure Yaml

Please choose a yaml config file containing the target pre-defined model and data pipeline that you want to re-use from `configs/det`. Here we choose `configs/det/dbnet/db_r50_icdar15.yaml`.

And change the data config args according to 
``` yaml
train:
  dataset:
    data_dir: PATH/TO/TRAIN_IMAGES_DIR
    label_file: PATH/TO/TRAIN_LABELS.txt
eval:
  dataset:
    data_dir: PATH/TO/TEST_IMAGES_DIR
    label_file: PATH/TO/TEST_LABELS.txt
```

Optionally, change `num_workers` according to the cores of CPU, and change `distribute` to True if you are to train in distributed mode.

#### 3. Training

To train the model, please run 

``` shell 
# train dbnet on ic15 dataset
python tools/train.py --config configs/det/dbnet/db_r50_icdar15.yaml
```

The training result (including checkpoints, per-epoch performance and curves) will be  saved in the directory parsed by the arg `ckpt_save_dir`.

#### 4. Evaluation

To evaluate, please set the checkpoint path to the arg `ckpt_load_path` in yaml config file and run 

``` shell
python tools/eval.py --config configs/det/dbnet/db_r50_icdar15.yaml
```


### Text Recognition Model Training

We will use **CRNN** model and **LMDB** dataset for illustration, although other models and datasets are also supported. 

#### 1. Data Preparation

Please download the LMDB dataset from [here](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0) (ref: [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here)).

There're several .zip data files:
- `data_lmdb_release.zip` contains the entire datasets including train, valid and evaluation.
- `validation.zip` is the union dataset for Validation
- `evaluation.zip` contains several benchmarking datasets. 

Unzip the data and after preparation, the data structure should be like 

``` text
.
├── train
│   ├── MJ
│   │   ├── data.mdb
│   │   ├── lock.mdb
│   ├── ST
│   │   ├── data.mdb
│   │   ├── lock.mdb
└── validation
|   ├── data.mdb
|   ├── lock.mdb
└── evaluation
    ├── IC03
    │   ├── data.mdb
    │   ├── lock.mdb
    ├── IC13
    │   ├── data.mdb
    │   ├── lock.mdb
    └── ...
```

#### 2. Configure Yaml

Please choose a yaml config file containing the target pre-defined model and data pipeline that you want to re-use from `configs/rec`. Here we choose `configs/rec/crnn/crnn_resnet34.yaml`.

Please change the data config args accordingly, such as
``` yaml
train:
  dataset:
    type: LMDBDataset
    data_dir: lmdb_data/rec/train/
eval:
  dataset:
    type: LMDBDataset
    data_dir: lmdb_data/rec/validation/
```

Optionally, change `num_workers` according to the cores of CPU, and change `distribute` to True if you are to train in distributed mode.

#### 3. Training

We will use distributed training for the large LMDB dataset. 

To train in distributed mode, please run

```shell
# Distributed training on Ascends
mpirun --allow-run-as-root -n 8 python tools/train.py  --config configs/rec/crnn/crnn_resnet34.yaml
```

```shell
# n is the number of GPUs/NPUs
mpirun --allow-run-as-root -n 2 python tools/train.py --config configs/rec/crnn/crnn_resnet34.yaml
```
> Notes: please ensure the arg `distribute` in yaml file is set True


The training result (including checkpoints, per-epoch performance and curves) will be  saved in the directory parsed by the arg `ckpt_save_dir`. 

#### 4. Evaluation

To evaluate, please set the checkpoint path to the arg `ckpt_load_path` in yaml config file and run 

``` shell
python tools/eval.py --config configs/rec/crnn/crnn_resnet34.yaml
```

### Inference and Deployment

#### Inference with MX Engine

Please refer to [mx_infer](docs/cn/inference_cn.md)

#### Inference with Lite 

Coming soon

#### Inference with native MindSpore

Coming soon

## Supported Models and Performance

### Text Detection  

The supported detection  models and their performance on the test set of ICDAR2015 are as follow.

| **Model** | **Backbone** | **Pretrained** | **Recall** | **Precision** | **F-score** | **Config**                                        | 
|-----------|--------------|----------------|------------|---------------|-------------|---------------------------------------------------|
| DBNet     | ResNet-50    | ImageNet       | 81.97%     | 86.05%        | 83.96%      | [YAML](configs/det/dbnet/db_r50_icdar15.yaml)     | 
| DBNet++   | ResNet-50    | ImageNet       | 82.02%     | 87.38%        | 84.62%      | [YAML](configs/det/dbnet++/db++_r50_icdar15.yaml) |

### Text Recognition

The supported recognition models and their overall performance on the public benchmarking datasets (IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE) are as follow


| **Model** | **Backbone** | **Avg Acc**| **Config** | 
|-----------|--------------|----------------|------------|
| CRNN     | VGG7        | 82.03% 	| [YAML](configs/rec/crnn/crnn_vgg7.yaml)    | 
| CRNN     | Resnet34_vd    | 84.45% 	| [YAML](configs/rec/crnn/crnn_resnet34.yaml)     |


## Notes

### Change Log
- 2023/03/23
1. Add dynamic loss scaler support, compatiable with drop overflow update. To enable dynamic loss scaler, please set `type` of `loss_scale` as `dynamic`. A yaml example can be viewed in `configs/rec/crnn/crnn_icdar15.yaml`

- 2023/03/20
1. Arg names changed: `output_keys` -> `output_columns`, `num_keys_to_net` -> `num_columns_to_net`
2. Data pipeline updated

- 2023/03/13
1. Add system test and CI workflow.
2. Add modelarts adapter to allow training on OpenI platform. To train on OpenI:
  ```text
    i)   Create a new training task on the openi cloud platform.
    ii)  Link the dataset (e.g., ic15_mindocr) on the webpage.
    iii) Add run parameter `config` and write the yaml file path on the website UI interface, e.g., '/home/work/user-job-dir/V0001/configs/rec/test.yaml'
    iv)  Add run parameter `enable_modelarts` and set True on the website UI interface.
    v)   Fill in other blanks and launch.
  ```

- 2023/03/08
1. Add evaluation script with  arg `ckpt_load_path`
2. Arg `ckpt_save_dir` is moved from `system` to `train` in yaml.
3. Add drop_overflow_update control

### How to Contribute

We appreciate all kind of contributions including issues and PRs to make MindOCR better.

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline. Please follow the [Model Template and Guideline](mindocr/models/README.md) for contributing a model that fits the overall interface :)

### License

This project follows the [Apache License 2.0](LICENSE.md) open-source license.

### Citation

If you find this project useful in your research, please consider citing:

```latex
@misc{MindSpore OCR 2023,
    title={{MindSpore OCR }:MindSpore OCR Toolbox},
    author={MindSpore Team},
    howpublished = {\url{https://github.com/mindspore-lab/mindocr/}},
    year={2023}
}
```
