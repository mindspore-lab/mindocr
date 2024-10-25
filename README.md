<!--start-->
<div align="center" markdown>

# MindOCR

</div>
<!--end-->

<div align="center" markdown>

[![CI](https://github.com/mindspore-lab/mindocr/actions/workflows/ci.yml/badge.svg)](https://github.com/mindspore-lab/mindocr/actions/workflows/ci.yml)
[![license](https://img.shields.io/github/license/mindspore-lab/mindocr.svg)](LICENSE)
[![open issues](https://img.shields.io/github/issues/mindspore-lab/mindocr)](https://github.com/mindspore-lab/mindocr/issues)
[![PRs](https://img.shields.io/badge/PRs-welcome-pink.svg)](https://github.com/mindspore-lab/mindocr/pulls)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

English | [中文](README_CN.md)

[📝Introduction](#introduction) |
[🔨Installation](#installation) |
[🚀Quick Start](#quick-start) |
[📚Tutorials](#tutorials) |
[🎁Model List](#model-list) |
[📰Dataset List](#dataset-list) |
[📖Frequently Asked Questions](#frequently-asked-questions) |
[🎉Notes](#notes)

</div>

<!--start-->
## Introduction
MindOCR is an open-source toolbox for OCR development and application based on [MindSpore](https://www.mindspore.cn/en), which integrates series of mainstream text detection and recognition algorihtms/models, provides easy-to-use training and inference tools. It can accelerate the process of developing and deploying SoTA text detection and recognition models in real-world applications, such as DBNet/DBNet++ and CRNN/SVTR, and help fulfill the need of image-text understanding.


<details open markdown>
<summary> Major Features </summary>

- **Modular design**: We decoupled the OCR task into several configurable modules. Users can setup the training and evaluation pipelines, customize the data processing pipeline and model architectures easily by modifying just few lines of code.
- **High-performance**: MindOCR provides a series of pretrained weights trained with optimized configurations that reach competitive performance on OCR tasks.
- **Low-cost-to-apply**: Easy-to-use inference tools are provided in MindOCR to perform text detection and recognition tasks.
</details>

The following is the corresponding `mindocr` versions and supported
mindspore versions.

| mindocr | mindspore |
|:-------:|:---------:|
| master  |  master   |
|   0.4   |   2.3.0   |
|   0.3   |  2.2.10   |
|   0.1   |    1.8    |


## Installation

<details open markdown>
<summary> Details </summary>

#### Prerequisites

MindOCR is built on MindSpore AI framework and is compatible with the following framework versions. installation guideline for Training, please refer to the installation links shown below.

- mindspore [[install](https://www.mindspore.cn/install)] Please install correct MindSpore version refer to `mindocr` versions.
- python >= 3.7
- openmpi 4.0.3 (for distributed training/evaluation)  [[install](https://www.open-mpi.org/software/ompi/v4.0/)]

MindSpore Lite offline Inference please refer to [Lite offline Environment Installation](docs/en/inference/environment.md)

#### Dependency
```shell
pip install -r requirements.txt
```

#### Install from Source (recommend)
```shell
git clone https://github.com/mindspore-lab/mindocr.git
cd mindocr
pip install -e .
```
> Using `-e` for "editable" mode can help resolve potential module import issues.

#### Install from docker

<details>
<summary> Details </summary>

The environment information of dockers provided is as following:
 - OS：Euler2.8
 - CANN：7.0
 - Python：3.9
 - MindSpore：2.2.10
 - MindSpore Lite：2.2.10

Please follow the steps to install docker：

1. Download docker
    - 910：
        ```bash
        docker pull swr.cn-central-221.ovaijisuan.com/mindocr/mindocr_dev_910_ms_2_2_10_cann7_0_py39:v1
        ```
    - 910*:
        ```bash
        docker pull swr.cn-central-221.ovaijisuan.com/mindocr/mindocr_dev_ms_2_2_10_cann7_0_py39:v1
        ```
2. Create container
    ```bash
    docker_name="temp_mindocr"
    # 910
    image_name="swr.cn-central-221.ovaijisuan.com/mindocr/mindocr_dev_910_ms_2_2_10_cann7_0_py39:v1"
    # 910*
    image_name="swr.cn-central-221.ovaijisuan.com/mindocr/mindocr_dev_ms_2_2_10_cann7_0_py39:v1"

    docker run --privileged --name ${docker_name} \
        --tmpfs /tmp \
        --tmpfs /run \
        -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
        --device=/dev/davinci1 \
        --device=/dev/davinci2 \
        --device=/dev/davinci3 \
        --device=/dev/davinci4 \
        --device=/dev/davinci5 \
        --device=/dev/davinci6 \
        --device=/dev/davinci7 \
        --device=/dev/davinci_manager \
        --device=/dev/hisi_hdc \
        --device=/dev/devmm_svm \
        -v /etc/localtime:/etc/localtime \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        --shm-size 800g \
        --cpus 96 \
        --security-opt seccomp=unconfined \
        --network=bridge -itd ${image_name} bash
    ```

3. Enter container
    ```bash
    # set docker id
    container_id="your docker id"
    docker exec -it --user root $container_id bash
    ```

4. Set environment variables
    After entering container, set environment variables by the following command：
    ```bash
    source env_setup.sh
    ```

</details>

#### Install from PyPI
```shell
pip install mindocr
```
> As this project is under active development, the version installed from PyPI is out-of-date currently. (will update soon).

</details>

## Quick Start

### 1. Text Detection and Recognition Demo

After installing MindOCR, we can run text detection and recognition on an arbitrary image easily as follows.

```shell
python tools/infer/text/predict_system.py --image_dir {path_to_img or dir_to_imgs} \
                                          --det_algorithm DB++  \
                                          --rec_algorithm CRNN  \
                                          --visualize_output True
```

After running, the results will be saved in `./inference_results` by default. Here is an example result.

<p align="center">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/c1f53970-8618-4039-994f-9f6dc1eee1dd" width=600 />
</p>
<p align="center">
  <em> Visualization of text detection and recognition result </em>
</p>

We can see that all texts on the image are detected and recognized accurately. For more usage, please refer to the inference section in [tutorials](#tutorials).

### 2. Model Training and Evaluation - Quick Guideline

It is easy to train your OCR model with the `tools/train.py` script, which supports both text detection and recognition model training.

```shell
python tools/train.py --config {path/to/model_config.yaml}
```

The `--config` arg specifies the path to a yaml file that defines the model to be trained and the training strategy including data process pipeline, optimizer, lr scheduler, etc.

MindOCR provides SoTA OCR models with their training strategies in `configs` folder.
You may adapt it to your task/dataset, for example, by running

```shell
# train text detection model DBNet++ on icdar15 dataset
python tools/train.py --config configs/det/dbnet/dbpp_r50_icdar15.yaml
```

```shell
# train text recognition model CRNN on icdar15 dataset
python tools/train.py --config configs/rec/crnn/crnn_icdar15.yaml
```

Similarly, it is easy to evaluate the trained model with the `tools/eval.py` script.

```shell
python tools/eval.py \
    --config {path/to/model_config.yaml} \
    --opt eval.dataset_root={path/to/your_dataset} eval.ckpt_load_path={path/to/ckpt_file}
```

For more illustration and usage, please refer to the model training section in [Tutorials](#tutorials).

### 3. Model Offline Inference

You can do MindSpore Lite inference in MindOCR using **MindOCR models** or **Third-party models** (PaddleOCR, MMOCR, etc.). Please refer to [Model Offline Inference Tutorial](docs/en/inference/inference_tutorial.md)

## Tutorials

- Datasets
    - [Dataset Preparation](docs/en/datasets/converters.md)
    - [Data Transformation Mechanism](docs/en/tutorials/transform_tutorial.md)
- Model Training
    - [Yaml Configuration](docs/en/tutorials/yaml_configuration.md)
    - [Text Detection](docs/en/tutorials/training_detection_custom_dataset.md)
    - [Text Recognition](docs/en/tutorials/training_recognition_custom_dataset.md)
    - [Distributed Training](docs/en/tutorials/distribute_train.md)
    - [Advance: Gradient Accumulation, EMA, Resume Training, etc](docs/en/tutorials/advanced_train.md)
- Inference with MindSpore
    - [Python Online Inference](tools/infer/text/README.md)
- Inference with MindSpore Lite
    - [Model Offline Inference Tutorial](docs/en/inference/inference_tutorial.md)
- Developer Guides
    - [Customize Dataset](mindocr/data/README.md)
    - [Customize Data Transformation](mindocr/data/transforms/README.md)
    - [Customize a New Model](mindocr/models/README.md)
    - [Customize Postprocessing Method](mindocr/postprocess/README.md)

## Model List

<details open markdown>
<summary>Text Detection</summary>

- [x] [DBNet](configs/det/dbnet/README.md) (AAAI'2020)
- [x] [DBNet++](configs/det/dbnet/README.md) (TPAMI'2022)
- [x] [PSENet](configs/det/psenet/README.md) (CVPR'2019)
- [x] [EAST](configs/det/east/README.md)(CVPR'2017)
- [x] [FCENet](configs/det/fcenet/README.md) (CVPR'2021)

</details>

<details open markdown>
<summary>Text Recognition</summary>

- [x] [CRNN](configs/rec/crnn/README.md) (TPAMI'2016)
- [x] [CRNN-Seq2Seq/RARE](configs/rec/rare/README.md) (CVPR'2016)
- [x] [SVTR](configs/rec/svtr/README.md) (IJCAI'2022)
- [x] [MASTER](configs/rec/master/README.md) (PR'2019)
- [x] [VISIONLAN](configs/rec/visionlan/README.md) (ICCV'2021)
- [x] [RobustScanner](configs/rec/robustscanner/README.md) (ECCV'2020)
- [x] [ABINet](configs/rec/abinet/README.md) (CVPR'2021)

</details>

<details open markdown>
<summary>Layout Analysis</summary>

- [x] [YOLOv8](configs/layout/yolov8/README.md) ([Ultralytics Inc.](https://github.com/ultralytics/ultralytics))

</details>

<details open markdown>
<summary>Key Information Extraction</summary>

- [x] [LayoutXLM](configs/kie/vi_layoutxlm/README.md) (arXiv'2021)
- [x] [LayoutLMv3](configs/kie/layoutlmv3/README.md) (arXiv'2022)

</details>

<details open markdown>
<summary>Table Recognition</summary>

- [x] [TableMaster](configs/table/README.md) (arXiv'2021)

</details>

<details open markdown>
<summary>OCR large model</summary>

- [x] [Vary](configs/llm/vary/README.md) (arXiv'2023)

</details>

For the detailed performance of the trained models, please refer to [https://github.com/mindspore-lab/mindocr/blob/main/configs](./configs).

For details of MindSpore Lite and ACL inference models support, please refer to [MindOCR Models Support List](docs/en/inference/mindocr_models_list.md) and [Third-party Models Support List](docs/en/inference/thirdparty_models_list.md) (PaddleOCR etc.).

## Dataset List

MindOCR provides a [dataset conversion tool](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters) to OCR datasets with different formats and support customized dataset by users. We have validated the following public OCR datasets in model training/evaluation.

<details close markdown>
<summary>General OCR Datasets</summary>

- [Born-Digital Images](https://rrc.cvc.uab.es/?ch=1) [[download](docs/en/datasets/borndigital.md)]
- [CASIA-10K](http://www.nlpr.ia.ac.cn/pal/CASIA10K.html) [[download](docs/en/datasets/casia10k.md)]
- [CCPD](https://github.com/detectRecog/CCPD) [[download](docs/en/datasets/ccpd.md)]
- [Chinese Text Recognition Benchmark](https://github.com/FudanVI/benchmarking-chinese-text-recognition) [[paper](https://arxiv.org/abs/2112.15093)] [[download](docs/en/datasets/chinese_text_recognition.md)]
- [COCO-Text](https://rrc.cvc.uab.es/?ch=5) [[download](docs/en/datasets/cocotext.md)]
- [CTW](https://ctwdataset.github.io/) [[download](docs/en/datasets/ctw.md)]
- [ICDAR2015](https://rrc.cvc.uab.es/?ch=4) [[paper](https://rrc.cvc.uab.es/files/short_rrc_2015.pdf)] [[download](docs/en/datasets/icdar2015.md)]
- [ICDAR2019 ArT](https://rrc.cvc.uab.es/?ch=14) [[download](docs/en/datasets/ic19_art.md)]
- [LSVT](https://rrc.cvc.uab.es/?ch=16) [[download](docs/en/datasets/lsvt.md)]
- [MLT2017](https://rrc.cvc.uab.es/?ch=8) [[paper](https://ieeexplore.ieee.org/abstract/document/8270168)] [[download](docs/en/datasets/mlt2017.md)]
- [MSRA-TD500](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500)) [[paper](https://ieeexplore.ieee.org/abstract/document/6247787)] [[download](docs/en/datasets/td500.md)]
- [MTWI-2018](https://tianchi.aliyun.com/competition/entrance/231651/introduction) [[download](docs/en/datasets/mtwi2018.md)]
- [RCTW-17](https://rctw.vlrlab.net/) [[download](docs/en/datasets/rctw17.md)]
- [ReCTS](https://rrc.cvc.uab.es/?ch=12) [[download](docs/en/datasets/rects.md)]
- [SCUT-CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector) [[paper](https://www.sciencedirect.com/science/article/pii/S0031320319300664)] [[download](docs/en/datasets/ctw1500.md)]
- [SROIE](https://rrc.cvc.uab.es/?ch=13) [[download](docs/en/datasets/sroie.md)]
- [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset) [[download](docs/en/datasets/svt.md)]
- [SynText150k](https://github.com/aim-uofa/AdelaiDet) [[paper](https://arxiv.org/abs/2002.10200)] [[download](docs/en/datasets/syntext150k.md)]
- [SynthText](https://www.robots.ox.ac.uk/~vgg/data/scenetext/) [[paper](https://www.robots.ox.ac.uk/~vgg/publications/2016/Gupta16/)] [[download](docs/en/datasets/synthtext.md)]
- [TextOCR](https://textvqa.org/textocr/) [[download](docs/en/datasets/textocr.md)]
- [Total-Text](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Dataset) [[paper](https://arxiv.org/abs/1710.10400)] [[download](docs/en/datasets/totaltext.md)]

</details>

<details close markdown>
<summary>Layout Analysis Datasets</summary>

- [PublayNet](https://github.com/ibm-aur-nlp/PubLayNet) [[paper](https://arxiv.org/abs/1908.07836)] [[download](https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/publaynet.tar.gz)]

</details>

<details close markdown>
<summary>Key Information Extraction Datasets</summary>

- [XFUND](https://github.com/doc-analysis/XFUND) [[paper](https://aclanthology.org/2022.findings-acl.253/)] [[download](https://github.com/doc-analysis/XFUND/releases/tag/v1.0)]

</details>

<details close markdown>
<summary>Table Recognition Datasets</summary>

- [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet) [[paper](https://arxiv.org/pdf/1911.10683.pdf)] [[download](https://dax-cdn.cdn.appdomain.cloud/dax-pubtabnet/2.0.0/pubtabnet.tar.gz)]

</details>

We will include more datasets for training and evaluation. This list will be continuously updated.

## Frequently Asked Questions
Frequently asked questions about configuring environment and mindocr, please refer to [FAQ](docs/en/tutorials/frequently_asked_questions.md).

## Notes

### What is New

<details close markdown>
<summary>News</summary>

- 2023/04/01
1. Add new trained models
    - [LayoutLMv3](configs/kie/layoutlmv3/) for key information extraction

- 2024/03/20
1. Add new trained models
    - [Vary-toy](configs/llm/vary/vary_toy.yaml) for OCR large model, providing Qwen-1.8B LLM-based object detection and OCR abilities

- 2023/12/25
1. Add new trained models
    - [TableMaster](configs/table/table_master.yaml) for table recognition
2. Add more benchmark datasets and their results
    - [PubTabNet](configs/table/README.md)

- 2023/12/14
1. Add new trained models
    - [LayoutXLM](configs/kie/layoutxlm) for key information extraction
    - [VI-LayoutXLM](configs/kie/vi_layoutxlm) for key information extraction
    - [PP-OCRv3 DBNet](configs/det/dbnet/db_mobilenetv3_ppocrv3.yaml) for text detection and [PP-OCRv3 SVTR](configs/rec/svtr/svtr_ppocrv3_ch.yaml) for recognition, supporting online inferece and finetuning
2. Add more benchmark datasets and their results
    - [XFUND](configs/kie/vi_layoutxlm/README_CN.md)
3. Multiple specifications support for Ascend 910: DBNet ResNet-50, DBNet++ ResNet-50, CRNN VGG7, SVTR-Tiny, FCENet, ABINet
- 2023/11/28
1. Add offline inference support for PP-OCRv4
    - [PP-OCRv4 DBNet](deploy/py_infer/src/configs/det/ppocr/ch_PP-OCRv4_det_cml.yaml) for text detection and [PP-OCRv4 CRNN](deploy/py_infer/src/configs/rec/ppocr/ch_PP-OCRv4_rec_distillation.yaml) for text recognition, supporting offline inferece
2. Fix bugs of third-party models offline inference
- 2023/11/17
1. Add new trained models
    - [YOLOv8](configs/layout/yolov8) for layout analysis
2. Add more benchmark datasets and their results
    - [PublayNet](configs/layout/yolov8/README_CN.md)
- 2023/07/06
1. Add new trained models
    - [RobustScanner](configs/rec/robustscanner) for text recognition
- 2023/07/05
1. Add new trained models
    - [VISIONLAN](configs/rec/visionlan) for text recognition
- 2023/06/29
1. Add new trained models
    - [FCENet](configs/det/fcenet) for text detection
    - [MASTER](configs/rec/master) for text recognition
- 2023/06/07
1. Add new trained models
    - [PSENet](configs/det/psenet) for text detection
    - [EAST](configs/det/east) for text detection
    - [SVTR](configs/rec/svtr) for text recognition
2. Add more benchmark datasets and their results
    - [totaltext](docs/en/datasets/totaltext.md)
    - [mlt2017](docs/en/datasets/mlt2017.md)
    - [chinese_text_recognition](docs/en/datasets/chinese_text_recognition.md)
3. Add resume training function, which can be used in case of unexpected interruption in training. Usage: add the `resume` parameter under the `model` field in the yaml config, e.g.,`resume: True`, load and resume training from {ckpt_save_dir}/train_resume.ckpt or `resume: /path/to/train_resume.ckpt`, load and resume training from the given path.
4. Improve postprocessing for detection: re-scale detected text polygons to original image space by default,
which can be enabled by add "shape_list" to the `eval.dataset.output_columns` list.
5. Refactor online inference to support more models, see [README.md](tools/infer/text/README.md) for details.

- 2023/05/15
1. Add new trained models
    - [DBNet++](configs/det/dbnet) for text detection
    - [CRNN-Seq2Seq](configs/rec/rare) for text recognition
    - DBNet pretrained on SynthText is now available: [checkpoint url](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_synthtext-40655acb.ckpt)
2. Add more benchmark datasets and their results
    - [SynthText](docs/en/datasets/synthtext.md), [MSRA-TD500](docs/en/datasets/td500.md), [CTW1500](docs/en/datasets/ctw1500.md)
    - More benchmark results for DBNet are reported [here](configs/det/dbnet/README.md).
3. Add checkpoint manager for saving top-k checkpoints and improve log.
4. Python inference code refactored.
5. Bug fix: use Meter to average loss for large datasets, disable `pred_cast_fp32` for ctcloss in AMP training, fix error when invalid polygons exist.

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
</details>

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
<!--end-->
