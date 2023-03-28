English | [中文](README_CN.md)

# DBNet

<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)

## Introduction

DBNet is a segmentation-based scene text detection method. Segmentation-based methods are gaining popularity for scene
text detection purposes as they can more accurately describe scene text of various shapes, such as curved text.  
The drawback of current segmentation-based SOTA methods is the post-processing of binarization (conversion of
probability maps into text bounding boxes) which often requires a manually set threshold (reduces prediction accuracy)
and complex algorithms for grouping pixels (resulting in a considerable time cost during inference).  
To eliminate the problem described above, DBNet integrates an adaptive threshold called Differentiable Binarization(DB)
into the architecture. DB simplifies post-processing and enhances the performance of text detection.Moreover, it can be
removed in the inference stage without sacrificing performance.[[1](#references)]

![dbnet_architecture](https://user-images.githubusercontent.com/16683750/225589619-d50c506c-e903-4f59-a316-8b62586c73a9.png)
<p align="center"><em>Figure 1. Overall DBNet architecture</em></p>

The overall architecture of DBNet is presented in _Figure 1._ It consists of multiple stages:

1. Feature extraction from a backbone at different scales. ResNet-50 is used as a backbone, and features are extracted
   from stages 2, 3, 4, and 5.
2. The extracted features are upscaled and summed up with the previous stage features in a cascade fashion.
3. The resulting features are upscaled once again to match the size of the largest feature map (from the stage 2) and
   concatenated along the channel axis.
4. Then, the final feature map (shown in dark blue) is used to predict both the probability and threshold maps by
   applying 3×3 convolutional operator and two de-convolutional operators with stride 2.
5. The probability and threshold maps are merged into one approximate binary map by the Differentiable binarization
   module. The approximate binary map is used to generate text bounding boxes.

## Results

### ICDAR2015
<div align="center">

| **Model** | **Backbone** | **Pretrained** | **Recall** | **Precision** | **F-score** | **Recipe**                         | **Download**                                                                                 |
|------------------|--------------|----------------|------------|---------------|-------------|-----------------------------|----------------------------------------------------------------------------------------------|
| DBNet (ours)     | ResNet-50    | ImageNet       | 81.70%     | 85.84%        | 83.72%      | [yaml](db_r50_icdar15.yaml) | [weights](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50-db1df47a.ckpt) |
| DBNet (PaddleOCR)| ResNet50_vd  | SynthText      | 78.72%     | 86.41%        | 82.38%      |

</div>

## Quick Start

### Preparation

#### Installation

Please refer to the [installation instruction](https://github.com/mindspore-lab/mindocr#installation) in MindOCR.

### Dataset preparation

First, the dataset labels need to be converted. To do so,
download [ICDAR2015](https://rrc.cvc.uab.es/?ch=4&com=downloads), and extract images and labels in a preferred folder.
Then, use the following command to generate _training_ labels:

```shell
python tools/dataset_converters/convert.py --dataset_name=ic15 --task=det --image_dir=IMAGES_DIR --label_dir=LABELS_DIR --output_path=OUTPUT_PATH
```

Repeat this step to generate the _test_ labels.

After the label files are generated, update `configs/det/dbnet/db_r50_icdar15.yaml` configuration file with data paths,
specifically the following parts:

```yaml
...
train:
  ckpt_save_dir: './tmp_det'
  dataset_sink_mode: True
  dataset:
    type: DetDataset
    dataset_root: /data/ocr_datasets                                  <------ HERE
    data_dir: ic15/text_localization/train                            <------ HERE
    label_file: ic15/text_localization/train/train_icdar15_label.txt  <------ HERE
...
eval:
  dataset_sink_mode: False
  dataset:
    type: DetDataset
    dataset_root: /data/ocr_datasets                                  <------ HERE
    data_dir: ic15/text_localization/test                             <------ HERE
    label_file: ic15/text_localization/test/test_icdar2015_label.txt  <------ HERE
...
```

### Config explanation

_DBNet_ consists of 3 parts: `backbone`, `neck`, and `head`. Specifically:

```yaml
model:
  type: det
  transform: null
  backbone:
    name: det_resnet50  # Only ResNet50 is supported at the moment
    pretrained: True    # Whether to use weights pretrained on ImageNet
  neck:
    name: DBFPN         # FPN part of the DBNet
    out_channels: 256
    bias: False
    use_asf: False      # Adaptive Scale Fusion module from DBNet++ (use it for DBNet++ only)
  head:
    name: DBHead
    k: 50               # amplifying factor for Differentiable Binarization
    bias: False
    adaptive: True      # True for training, False for inference
```

[comment]: <> (The only difference between _DBNet_ and _DBNet++_ is in the _Adaptive Scale Fusion_ module, which is controlled by the `use_asf` parameter in the `neck` module.)

### Training

After preparing a dataset and setting the configuration, training can be started as follows:

```shell
python tools/train.py -c=configs/det/dbnet/db_r50_icdar15.yaml
```

### Evaluation

To evaluate the accuracy of the trained model, you can use `eval.py`. Please set the checkpoint path to the arg `ckpt_load_path` in the `eval` section of yaml config file and then run:

```shell
python tools/eval.py -c=configs/det/dbnet/db_r50_icdar15.yaml
```

## References

<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Minghui Liao, Zhaoyi Wan, Cong Yao, Kai Chen, Xiang Bai. Real-time Scene Text Detection with Differentiable Binarization. arXiv:1911.08947, 2019
