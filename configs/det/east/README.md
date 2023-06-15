English | [中文](README_CN.md)

# EAST

<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155)

## 1. Introduction

### EAST

EAST (Efficient and Accurate Scene Text Detection) is an efficient, accurate, and lightweight OCR detection algorithm mainly used for text detection in natural scenes. The algorithm extracts text features using deep residual networks, fuses features in a feature pyramid network, and adopts two branches for binary classification and text location detection. EAST has achieved significant results in the accuracy and robustness of text detection.

<p align="center"><img alt="Figure 1. east_architecture" src="https://github.com/tonytonglt/mindocr/assets/54050944/4781c9aa-64a5-4963-bf02-6620d173dc9a" width="384"/></p>
<p align="center"><em>Figure 1. Overall EAST architecture (We use ResNet-50 instead of PVANet in fig. above)</em></p>

The overall architecture of EAST is shown in Figure 1 and includes the following stages:

1. **Feature extraction**:
EAST uses Resnet-50 as the backbone network, feature extraction is performed at different levels, which are stages 2, 3, 4, and 5.

2. **Feature fusion**:
The features from different levels of the backbone network are fused in feature fusion stage. The feature maps are enlarged and connected along the channel axis to handle different text regions of varying sizes and improve detection accuracy.

3. **Boundary box regression**:
EAST uses regression for the position and rotation angle of the text box, enabling detection of inclined text to perform better of text detection tasks in natural scenes. Currently, the detection of rotated rectangles is supported.

4. **Text detection branch**:
After determining the location and size of the text region, EAST further classifies these regions as text or non-text areas. For this purpose, a fully convolutional text branch is employed for binary classification of the text areas.


## 2. Results

### ICDAR2015
<div align="center">

| **Model**           | **Context**       | **Backbone**    | **Pretrained** | **Recall** | **Precision** | **F-score** | **Train T.**    | **Throughput**   | **Recipe**                   | Download                                                                                                     |
|------------------|----------------|-------------|------------|------------|---------------|-------------|-------------|-----------|----------------------------|------------------------------------------------------------------------------------------------------------|
| EAST             | D910x8-MS1.9-G | ResNet-50   | ImageNet   | 82.23%     | 87.68%        | 84.87%      | 1.6 s/epoch | 625 img/s | [yaml](east_r50_icdar15.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/east/east_resnet50_ic15-7262e359.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/east/east_resnet50_ic15-7262e359-5f05cd42.mindir) |

</div>

#### Notes：
- Context: Training context denoted as {device}x{pieces}-{MS version}{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- The training time of EAST is highly affected by data processing and varies on different machines.
- The input_shape for exported MindIR in the link is `(1,3,720,1280)`.

## 3. Quick Start

### 3.1 Installation

Please refer to the [installation instruction](https://github.com/mindspore-lab/mindocr#installation) in MindOCR.

### 3.2 Dataset preparation

#### 3.2.1 ICDAR2015 dataset

Please download [ICDAR2015](https://rrc.cvc.uab.es/?ch=4&com=downloads) dataset, and convert the labels to the desired format referring to [dataset_converters](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters/README.md).

The prepared dataset file struture should be:

``` text
.
├── test
│   ├── images
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   └── ...
│   └── test_det_gt.txt
└── train
    ├── images
    │   ├── img_1.jpg
    │   ├── img_2.jpg
    │   └── ....jpg
    └── train_det_gt.txt
```

### 3.3 Update yaml config file

Update `configs/det/east/east_r50_icdar15.yaml` configuration file with data paths,
specifically the following parts. The `dataset_root` will be concatenated with `data_dir` and `label_file` respectively to be the complete dataset directory and label file path.

```yaml
...
train:
  ckpt_save_dir: './tmp_det'
  dataset_sink_mode: False
  dataset:
    type: DetDataset
    dataset_root: dir/to/dataset          <--- Update
    data_dir: train/images                <--- Update
    label_file: train/train_det_gt.txt    <--- Update
...
eval:
  dataset_sink_mode: False
  dataset:
    type: DetDataset
    dataset_root: dir/to/dataset          <--- Update
    data_dir: test/images                 <--- Update
    label_file: test/test_det_gt.txt      <--- Update
...
```

> Optionally, change `num_workers` according to the cores of CPU.


EAST consists of 3 parts: `backbone`, `neck`, and `head`. Specifically:

```yaml
model:
  type: det
  transform: null
  backbone:
    name: det_resnet50
    pretrained: True    # Whether to use weights pretrained on ImageNet
  neck:
    name: EASTFPN       # FPN part of the EAST
    out_channels: 128
  head:
    name: EASTHead
```

### 3.4 Training

* Standalone training

Please set `distribute` in yaml config file to be False.

``` shell
# train east on ic15 dataset
python tools/train.py --config configs/det/east/east_r50_icdar15.yaml
```

* Distributed training

Please set `distribute` in yaml config file to be True.

```shell
# n is the number of GPUs/NPUs
mpirun --allow-run-as-root -n 8 python tools/train.py --config configs/det/east/east_r50_icdar15.yaml
```

The training result (including checkpoints, per-epoch performance and curves) will be saved in the directory parsed by the arg `ckpt_save_dir` in yaml config file. The default directory is `./tmp_det`.

### 3.5 Evaluation

To evaluate the accuracy of the trained model, you can use `eval.py`. Please set the checkpoint path to the arg `ckpt_load_path` in the `eval` section of yaml config file, set `distribute` to be False, and then run:

``` shell
python tools/eval.py --config configs/det/east/east_r50_icdar15.yaml
```

### 3.6 MindSpore Lite Inference

Please refer to the tutorial [MindOCR Inference](../../../docs/en/inference/inference_tutorial_en.md) for model inference based on MindSpot Lite on Ascend 310, including the following steps:

- Model Export

Please [download](#2-results) the exported MindIR file first, or refer to the [Model Export](../../README.md) tutorial and use the following command to export the trained ckpt model to  MindIR file:

``` shell
python tools/export.py --model_name east_resnet50 --data_shape 720 1280 --local_ckpt_path /path/to/local_ckpt.ckpt
# or
python tools/export.py --model_name configs/det/east/east_r50_icdar15.yaml --data_shape 720 1280 --local_ckpt_path /path/to/local_ckpt.ckpt
```

The `data_shape` is the model input shape of height and width for MindIR file. The shape value of MindIR in the download link can be found in [Notes](#notes).

- Environment Installation

Please refer to [Environment Installation](../../../docs/en/inference/environment_en.md#2-mindspore-lite-inference) tutorial to configure the MindSpore Lite inference environment.

- Model Conversion

Please refer to [Model Conversion](../../../docs/en/inference/convert_tutorial_en.md#1-mindocr-models),
and use the `converter_lite` tool for offline conversion of the MindIR file, where the `input_shape` in `configFile` needs to be filled in with the value from MindIR export,
as mentioned above (1, 3, 720, 1280), and the format is NCHW.

- Inference

Assuming that you obtain output.mindir after model conversion, go to the `deploy/py_infer` directory, and use the following command for inference:

```shell
python infer.py \
    --input_images_dir=/your_path_to/test_images \
    --device=Ascend \
    --device_id=0 \
    --det_model_path=your_path_to/output.mindir \
    --det_model_name_or_config=../../configs/det/east/east_r50_icdar15.yaml \
    --backend=lite \
    --res_save_dir=results_dir
```

## References

<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Xinyu Zhou, Cong Yao, He Wen, Yuzhi Wang, Shuchang Zhou, Weiran He, Jiajun Liang. EAST: An Efficient and Accurate Scene Text Detector. arXiv:1704.03155, 2017
