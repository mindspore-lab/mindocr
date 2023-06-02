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
- The training time of EAST is highly affected by data processing and varies on different machines。

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

## References

<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Xinyu Zhou, Cong Yao, He Wen, Yuzhi Wang, Shuchang Zhou, Weiran He, Jiajun Liang. EAST: An Efficient and Accurate Scene Text Detector. arXiv:1704.03155, 2017
