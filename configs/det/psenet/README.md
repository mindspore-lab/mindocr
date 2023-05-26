English | [中文](README_CN.md)

# PSENet

<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> PSENet: [Shape Robust Text Detection With Progressive Scale Expansion Network](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Shape_Robust_Text_Detection_With_Progressive_Scale_Expansion_Network_CVPR_2019_paper.html)  

## 1. Introduction

### PSENet

PSENet is a text detection algorithm based on semantic segmentation. It can precisely locate text instances with arbitrary shapes, while most anchor-based algorithms cannot be used to detect text instances with arbitrary shapes. Also, two texts that are close to each other may cause the model to make wrong predictions. Therefore, in order to solve the above problems, PSENet also proposes a Progressive Scale Expansion (PSE) algorithm, which can successfully identify adjacent text instances[[1](#references)]。

<p align="center"><img alt="Figure 1. Overall PSENet architecture" src="https://github.com/VictorHe-1/mindocr_pse/assets/80800595/6ed1b691-52c4-4025-b256-a022aa5ef582" width="800"/></p>
<p align="center"><em>Figure 1. Overall PSENet architecture</em></p>

The overall architecture of PSENet is presented in Figure 1. It consists of multiple stages:

1. Feature extraction from a backbone at different scales. ResNet is used as a backbone, and features are extracted from stages 2, 3, 4 and 5.
2. The FPN network will then use the extracted features to produce new features of different scales and then concatenate them.
3. Use the features of the second stage to generate the final segmentation result using the PSE algorithm, and generate text bounding boxes.


## 2. Results

### ICDAR2015
<div align="center">

| **Model**              | **Context**       | **Backbone**      | **Pretrained** | **Recall** | **Precision** | **F-score** | **Train T.**     | **Throughput**   | **Recipe**                            | **Download**                                                                                                                                                                                                |
|---------------------|----------------|---------------|------------|------------|---------------|-------------|--------------|-----------|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PSENet               | D910x8-MS2.0-G | ResNet-152   | ImageNet   | 79.39%     | 84.83%        | 82.02%      | 138 s/epoch   | 7.57 img/s | [yaml](pse_r152_icdar15.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_resnet152_ic15-6058a798.ckpt) 
</div>

#### Notes：
- Context：Training context denoted as {device}x{pieces}-{MS version}{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- The training time of PSENet is highly affected by data processing and varies on different machines。

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

Update `configs/det/psenet/pse_r152_icdar15.yaml` configuration file with data paths,
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


PSENet consists of 3 parts: `backbone`, `neck`, and `head`. Specifically:

```yaml
model:
  type: det
  transform: null
  backbone:
    name: det_resnet152  
    pretrained: True    # Whether to use weights pretrained on ImageNet
  neck:
    name: PSEFPN         # FPN part of the PSENet
    out_channels: 128
  head:
    name: PSEHead
    hidden_size: 256
    out_channels: 7     # number of kernels
```

### 3.4 Training
* Postprocess

Before training, please make sure to compile the postprocessing codes in the /mindocr/postprocess/pse directory as follows:

``` shell 
python3 setup.py build_ext --inplace
```

* Standalone training

Please set `distribute` in yaml config file to be False.

``` shell 
# train psenet on ic15 dataset
python tools/train.py --config configs/det/psenet/pse_r152_icdar15.yaml
```

* Distributed training

Please set `distribute` in yaml config file to be True.

```shell
# n is the number of GPUs/NPUs
mpirun --allow-run-as-root -n 8 python tools/train.py --config configs/det/psenet/pse_r152_icdar15.yaml
```

The training result (including checkpoints, per-epoch performance and curves) will be saved in the directory parsed by the arg `ckpt_save_dir` in yaml config file. The default directory is `./tmp_det`. 

### 3.5 Evaluation

To evaluate the accuracy of the trained model, you can use `eval.py`. Please set the checkpoint path to the arg `ckpt_load_path` in the `eval` section of yaml config file, set `distribute` to be False, and then run: 

``` shell
python tools/eval.py --config configs/det/psenet/pse_r152_icdar15.yaml
```

## References

<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Wang, Wenhai, et al. "Shape robust text detection with progressive scale expansion network." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.