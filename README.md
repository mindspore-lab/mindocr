

# MindOCR

<!-- English | [中文](README_CN.md) -->

[Introduction](#introduction) |
[Installation](#installation) |
[Quick Start](#quick-start) |
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

It s recommended to install MindSpore following the official [instructions](https://www.mindspore.cn/install) for the best fit of your machine. To enable training in distributed mode, please also install [openmpi](https://www.open-mpi.org/software/ompi/v4.0/).


### Install with PyPI

Coming soon
```shell
 pip install mindocr
```

### Install from Source

The latest version of MindOCR can be installed as follows:
```shell
pip install git+https://github.com/mindspore-lab/mindocr.git
```

> Notes: MindOCR is only tested on Linux on GPU/Ascend devices currently.

## Quick Start

### Text Detection Model Training

We will use **DBNet** model and **ICDAR2015** dataset for illustration, although other models and datasets are also supported. <!--ICDAR15 is a commonly-used model and a benchmark for scene text recognition.-->

#### 1. Data Preparation

Please download the ICDAR2015 dataset from this [website](https://rrc.cvc.uab.es/?ch=4&com=downloads), then format the dataset annotation refer to [dataset_convert](tools/dataset/README.md).

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

Please choose a yaml config file containing the target pre-defined model and data pipeline that you want to re-use from `configs/det`. Here we choose `configs/det/db_r50_icdar15.yaml`.

Please change the data config args accordingly, such as
``` yaml
train:
  dataset:
    data_dir: ic15/det/train/images
    label_files: ic15/det/train/det_gt.txt
```

Optionally, change `num_workers` according to the cores of CPU, and change `distribute` to True if you are to train in distributed mode.

#### 3. Training

To train the model, please run 

``` shell 
python tools/train.py --config configs/det/db_r50_icdar15.yaml
```

To train in distributed mode, please run

```shell
# n is the number of GPUs/NPUs
mpirun --allow-run-as-root -n 2 python tools/train.py --config configs/det/db_r50_icdar15.yaml
```
> Notes: please ensure the arg `distribute` in yaml file is set True


The training result (including checkpoints, per-epoch performance and curves) will be  saved in the directory parsed by the arg `ckpt_save_dir`, which is "./tmp_det/" by default. 

#### 4. Evaluation

To evaluate, please parse the checkpoint path to the arg `ckpt_load_path` in yaml config file and run 

``` shell
python tools/eval.py --config configs/det/db_r50_icdar15.yaml
```


### Text Recognition Model Training

We will use **CRNN** model and **LMDB** dataset for illustration, although other models and datasets are also supported. 

#### 1. Data Preparation

Please download the LMDB dataset from ... 

After preparation, the data structure should be like 

``` text
```

#### 2. Configure Yaml

Please choose a yaml config file containing the target pre-defined model and data pipeline that you want to re-use from `configs/det`. Here we choose `configs/det/vgg7_bilistm_ctc.yaml`.

Please change the data config args accordingly, such as
``` yaml
train:
  dataset:
    data_dir: ic15/det/train/images
    label_files: ic15/det/train/det_gt.txt
```

Optionally, change `num_workers` according to the cores of CPU, and change `distribute` to True if you are to train in distributed mode.

#### 3. Training

To train the model, please run 

``` shell 
python tools/train.py --config configs/rec/vgg7_bilstm_ctc.py
```

To train in distributed mode, please run

```shell
# n is the number of GPUs/NPUs
mpirun --allow-run-as-root -n 2 python tools/train.py --config configs/det/vgg7_bilstm_ctc.yaml
```
> Notes: please ensure the arg `distribute` in yaml file is set True


The training result (including checkpoints, per-epoch performance and curves) will be  saved in the directory parsed by the arg `ckpt_save_dir`, which is "./tmp_det/" by default. 

#### 4. Evaluation

To evaluate, please parse the checkpoint path to the arg `ckpt_load_path` in yaml config file and run 

``` shell
python tools/eval.py --config /path/to/config.yaml
```

### Inference and Deployment

#### Inference with MX Engine

Please refer to [mx_infer](]deploy/mx_infer/README.md)

#### Inference with Lite 

Coming soon

#### Inference with native MindSpore

Coming soon

## Notes

### Change Log

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
