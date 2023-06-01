
<div align="center">

# MindOCR

[![CI](https://github.com/mindspore-lab/mindocr/actions/workflows/ci.yml/badge.svg)](https://github.com/mindspore-lab/mindocr/actions/workflows/ci.yml)
[![license](https://img.shields.io/github/license/mindspore-lab/mindocr.svg)](https://github.com/mindspore-lab/mindocr/blob/main/LICENSE)
[![open issues](https://img.shields.io/github/issues/mindspore-lab/mindocr)](https://github.com/mindspore-lab/mindocr/issues)
[![PRs](https://img.shields.io/badge/PRs-welcome-pink.svg)](https://github.com/mindspore-lab/mindocr/pulls)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


English | [中文](README_CN.md)

[Introduction](#introduction) |
[Installation](#installation) |
[Quick Start](#quick-start) |
[Model List](#model-list) |
[Notes](#notes)

</div>


## Introduction
MindOCR is an open-source toolbox for OCR development and application based on [MindSpore](https://www.mindspore.cn/en). It helps users to train and apply the best text detection and recognition models, such as DBNet/DBNet++ and CRNN/SVTR, to fulfill image-text understanding needs.


<details open>
<summary> Major Features </summary>

- **Modulation design**: We decouple the OCR task into several configurable modules. Users can set up the training and evaluation pipeline easily for customized data and models with a few lines of modification.
- **High-performance**: MindOCR provides pretrained weights and the used training recipes that reach competitive performance on OCR tasks.
- **Low-cost-to-apply**: We provide easy-to-use inference tools to perform text detection and recognition tasks. 
</details>


## Installation

### Dependency

To install the dependency, please run
```shell
pip install -r requirements.txt
```

Additionally, please install MindSpore(>=1.9) following the official [installation instructions](https://www.mindspore.cn/install) for the best fit of your machine. 

For distributed training, please install [openmpi 4.0.3](https://www.open-mpi.org/software/ompi/v4.0/).

| Environment | Version |
|-------------|---------|
| MindSpore   | >=1.9   |
| Python      | >=3.7   |

> Notes: 
> - If you [use ACL for Inference](#21-inference-with-mindspore-lite-and-acl), the version of Python should be 3.9.
> - If scikit_image cannot be imported, you can use the following command line to set environment variable `$LD_PRELOAD` referring to [here](https://github.com/opencv/opencv/issues/14884). Change `path/to` to your directory.
>   ```shell
>   export LD_PRELOAD=path/to/scikit_image.libs/libgomp-d22c30c5.so.1.0.0:$LD_PRELOAD
>   ```


### Install with PyPI

Coming soon

### Install from Source

The latest version of MindOCR can be installed as follows:
```shell
pip install git+https://github.com/mindspore-lab/mindocr.git
```

> Notes: MindOCR is only tested on MindSpore>=1.9, Linux on GPU/Ascend devices currently.

## Quick Start

### 1. Model Training and Evaluation

#### 1.1 Text Detection

We will take **DBNet** model and **ICDAR2015** dataset as an example to illustrate how to configure the training process with a few lines of modification on the yaml file.

Please refer to [DBNet Readme](configs/det/dbnet/README.md#3-quick-start) for detailed instructions.


#### 1.2 Text Recognition 

We will take **CRNN** model and **LMDB** dataset as an illustration on how to configure and launch the training process easily. 

Detailed instructions can be viewed in [CRNN Readme](configs/rec/crnn/README.md#3-quick-start).

**Note:**
The training pipeline is fully extendable. To train other text detection/recognition models on a new dataset, please configure the model architecture (backbone, neck, head) and data pipeline in the yaml file and launch the training script with `python tools/train.py -c /path/to/yaml_config`.

### 2. Inference and Deployment

#### 2.1 Inference with MindSpore Lite and ACL on Ascend 310 

MindOCR supports OCR model inference with [MindSpore Lite](https://www.mindspore.cn/lite/en) and [ACL](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/aclcppdevg/aclcppdevg_000004.html) (Ascend Computation Language)  backends. It integrates efficient text detection, classification and recognition inference pipeline for deployment. 

Please refer to [MindOCR Inference on Ascend 310](docs/en/inference/inference_tutorial_en.md) for detailed illustrations.

#### 2.2 Inference with native MindSpore on Ascend910/GPU/CPU

MindOCR provides easy-to-use text detection and recognition inference tools supporting CPU/GPU/Ascend910 devices, based on the MindOCR-trained models. 

Please refer to [MindOCR Online Inference](tools/infer/text/README.md) for details.


## Model List

<details open>
<summary>Text Detection</summary>

- [x] [DBNet](configs/det/dbnet/README.md) (AAAI'2020) 
- [x] [DBNet++](configs/det/dbnet/README.md) (TPAMI'2022)
- [x] [PSENet](configs/det/psenet/README.md) (CVPR'2019)
- [x] [EAST](configs/det/east/README.md)(CVPR'2017)
- [ ] [FCENet](https://arxiv.org/abs/2104.10442) (CVPR'2021) [coming soon]

</details>

<details open>
<summary>Text Recognition</summary>

- [x] [CRNN](configs/rec/crnn/README.md) (TPAMI'2016)
- [x] [CRNN-Seq2Seq/RARE](configs/rec/rare/README.md) (CVPR'2016)
- [x] [SVTR](configs/rec/svtr/README.md) (IJCAI'2022) 
- [ ] [ABINet](https://arxiv.org/abs/2103.06495) (CVPR'2021) [coming soon]


For the detailed performance of the trained models, please refer to [configs](./configs).

For detailed support for MindSpore Lite and ACL inference models, please refer to [MindOCR Models Support List](docs/en/inference/models_list_en.md) and [Third-Party Models Support List](docs/en/inference/models_list_thirdparty_en.md).

## Datasets

### Download 

We give instructions on how to download the following datasets.

<details open>
<summary>Text Detection</summary>

- [x] ICDAR2015 [paper](https://rrc.cvc.uab.es/files/short_rrc_2015.pdf) [homepage](https://rrc.cvc.uab.es/?ch=4) [download instruction](docs/en/datasets/icdar2015.md)

- [x] Total-Text [paper](https://arxiv.org/abs/1710.10400) [homepage](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Dataset) [download instruction](docs/en/datasets/totaltext.md)

- [x] Syntext150k [paper](https://arxiv.org/abs/2002.10200) [homepage](https://github.com/aim-uofa/AdelaiDet) [download instruction](docs/en/datasets/syntext150k.md)

- [x] MLT2017 [paper](https://ieeexplore.ieee.org/abstract/document/8270168) [homepage](https://rrc.cvc.uab.es/?ch=8&com=introduction) [download instruction](docs/en/datasets/mlt2017.md)

- [x] MSRA-TD500 [paper](https://ieeexplore.ieee.org/abstract/document/6247787) [homepage](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500)) [download instruction](docs/en/datasets/td500.md)

- [x] SCUT-CTW1500 [paper](https://www.sciencedirect.com/science/article/pii/S0031320319300664) [homepage](https://github.com/Yuliang-Liu/Curve-Text-Detector) [download instruction](docs/en/datasets/ctw1500.md)

</details>

### Conversion

After downloading these datasets in the `DATASETS_DIR` folder, you can run `bash tools/convert_datasets.sh` to convert all downloaded datasets into the target format. [Here](tools/dataset_converters/README.md) is an example of icdar2015 dataset converting.

## Notes

### Change Log
- 2023/05/15
1. Add new trained models
    - [DBNet++](configs/det/dbnet) for text detection
    - [CRNN-Seq2Seq](configs/rec/rare) for text recognition
    - DBNet pretrained on SynthText is now available: [checkpoint url](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_synthtext-40655acb.ckpt)
2. Add more benchmark datasets and their results 
    - [SynthText](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c), [MSRA-TD500](docs/en/datasets/td500.md), [CTW1500](docs/en/datasets/ctw1500.md) 
    - More benchmark results for DBNet are reported [here](configs/det/dbnet/README.md).
3. Add checkpoint manager for saving top-k checkpoints and improve log.
4. Python inference code refractored. 
5. Bug fix: use meter to average loss for large datasets, disable `pred_cast_fp32` for ctcloss in AMP training, fix error when invalid polygons exist.

- 2023/05/04
1. Support loading self-defined pretrained checkpoints via setting `model-pretrained` with checkpoint url or local path in yaml. 
2. Support setting probability for executing augmentation including rotation and flip.
3. Add Exponential Moving Average(EMA) for model training, which can be enabled by setting `train-ema` (default: False) and `train-ema_decay` in the yaml config. 
4. Arg parameter changed：`num_columns_to_net` -> `net_input_column_index`: change the column number feeding into the network to the column index.
5. Arg parameter changed：`num_columns_of_labels` -> `label_column_index`: change the column number corresponds to the label to the column index.

- 2023/04/21
1. Add parameter grouping to support flexible regularization in training. Usage: add `grouping_strategy` argument in yaml config to select a predefined grouping strategy, or use `no_weight_decay_params` argument to pick layers to exclude from weight decay (e.g., bias, norm). Example can be referred in `configs/rec/crnn/crnn_icdar15.yaml` 
2. Add gradient accumulation to support large batch size training. Usage: add `gradient_accumulation_steps` in yaml config, the global batch size = batch_size * devices * gradient_accumulation_steps. Example can be referred in `configs/rec/crnn/crnn_icdar15.yaml`
3. Add gradient clip to support training stablization. Enable it by setting `grad_clip` as True in yaml config.

- 2023/03/23
1. Add dynamic loss scaler support, compatible with drop overflow update. To enable dynamic loss scaler, please set `type` of `loss_scale` as `dynamic`. A YAML example can be viewed in `configs/rec/crnn/crnn_icdar15.yaml`

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

We appreciate all kinds of contributions including issues and PRs to make MindOCR better.

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline. Please follow the [Model Template and Guideline](mindocr/models/README.md) for contributing a model that fits the overall interface :)

### License

This project follows the [Apache License 2.0](LICENSE) open-source license.

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
