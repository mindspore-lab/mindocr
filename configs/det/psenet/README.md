English | [中文](README_CN.md)

# PSENet

<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> PSENet: [Shape Robust Text Detection With Progressive Scale Expansion Network](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Shape_Robust_Text_Detection_With_Progressive_Scale_Expansion_Network_CVPR_2019_paper.html)

## Introduction

### PSENet

PSENet is a text detection algorithm based on semantic segmentation. It can precisely locate text instances with arbitrary shapes, while most anchor-based algorithms cannot be used to detect text instances with arbitrary shapes. Also, two texts that are close to each other may cause the model to make wrong predictions. Therefore, in order to solve the above problems, PSENet also proposes a Progressive Scale Expansion (PSE) algorithm, which can successfully identify adjacent text instances[[1](#references)]。

<p align="center"><img alt="Figure 1. Overall PSENet architecture" src="https://github.com/VictorHe-1/mindocr_pse/assets/80800595/6ed1b691-52c4-4025-b256-a022aa5ef582" width="800"/></p>
<p align="center"><em>Figure 1. Overall PSENet architecture</em></p>

The overall architecture of PSENet is presented in Figure 1. It consists of multiple stages:

1. Feature extraction from a backbone at different scales. ResNet is used as a backbone, and features are extracted from stages 2, 3, 4 and 5.
2. The FPN network will then use the extracted features to produce new features of different scales and then concatenate them.
3. Use the features of the second stage to generate the final segmentation result using the PSE algorithm, and generate text bounding boxes.

## Requirements

| mindspore  | ascend driver  |    firmware    | cann toolkit/kernel |
|:----------:|:--------------:|:--------------:|:-------------------:|
|   2.3.1    |    24.1.RC2    |  7.3.0.1.231   |    8.0.RC2.beta1    |

## Quick Start

### Installation

Please refer to the [installation instruction](https://github.com/mindspore-lab/mindocr#installation) in MindOCR.

### Dataset preparation

#### ICDAR2015 dataset

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

#### SCUT-CTW1500 dataset

Please download [SCUT-CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector) dataset and convert the labels to the desired format referring to [dataset_converters](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters/README.md).

The prepared dataset file struture should be:

```txt
ctw1500
 ├── test_images
 │   ├── 1001.jpg
 │   ├── 1002.jpg
 │   ├── ...
 ├── train_images
 │   ├── 0001.jpg
 │   ├── 0002.jpg
 │   ├── ...
 ├── test_det_gt.txt
 ├── train_det_gt.tx
```

### Update yaml config file

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

### Training
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

### Evaluation

To evaluate the accuracy of the trained model, you can use `eval.py`. Please set the checkpoint path to the arg `ckpt_load_path` in the `eval` section of yaml config file, set `distribute` to be False, and then run:

``` shell
python tools/eval.py --config configs/det/psenet/pse_r152_icdar15.yaml
```

### MindSpore Lite Inference

Please refer to the tutorial [MindOCR Inference](../../../docs/en/inference/inference_tutorial.md) for model inference based on MindSpot Lite on Ascend 310, including the following steps:

- Model Export

Please [download](#2-results) the exported MindIR file first, or refer to the [Model Export](../../README.md) tutorial and use the following command to export the trained ckpt model to  MindIR file:

```shell
python tools/export.py --model_name_or_config psenet_resnet152 --data_shape 1472 2624 --local_ckpt_path /path/to/local_ckpt.ckpt
# or
python tools/export.py --model_name_or_config configs/det/psenet/pse_r152_icdar15.yaml --data_shape 1472 2624 --local_ckpt_path /path/to/local_ckpt.ckpt
```

The `data_shape` is the model input shape of height and width for MindIR file. The shape value of MindIR in the download link can be found in [Notes](#notes).

- Environment Installation

Please refer to [Environment Installation](../../../docs/en/inference/environment.md#2-mindspore-lite-inference) tutorial to configure the MindSpore Lite inference environment.

- Model Conversion

Please refer to [Model Conversion](../../../docs/en/inference/convert_tutorial.md#1-mindocr-models),
and use the `converter_lite` tool for offline conversion of the MindIR file.

- Inference

Before inference, please ensure that the post-processing part of PSENet has been compiled (refer to the post-processing part of the [Training](#34-training) chapter).

Assuming that you obtain output.mindir after model conversion, go to the `deploy/py_infer` directory, and use the following command for inference:

```shell
python infer.py \
    --input_images_dir=/your_path_to/test_images \
    --det_model_path=your_path_to/output.mindir \
    --det_model_name_or_config=../../configs/det/psenet/pse_r152_icdar15.yaml \
    --res_save_dir=results_dir
```

## Performance

PSENet were trained on the ICDAR2015, SCUT-CTW1500 datasets. In addition, we conducted pre-training on the ImageNet dataset and provided a URL to download pretrained weights. All training results are as follows:

### ICDAR2015

| **model name** | **backbone**  | **pretrained** | **cards** | **batch size** | **jit level** | **graph compile** | **ms/step** | **img/s** | **recall** | **precision** | **f-score** |               **recipe**               |                                             **weight**                                            |
|:--------------:|:-------------:| :------------: |:---------:|:--------------:| :-----------: |:-----------------:|:-----------:|:---------:|:----------:|:-------------:|:-----------:|:--------------------------------------:|:-------------------------------------------------------------------------------------------------:|
|     PSENet     |  ResNet-152   |    ImageNet    |     8     |       8        |      O2       |     225.02 s      |   355.19    |  180.19   |   78.91%   |    84.70%     |   81.70%    |    [yaml](pse_r152_icdar15.yaml)       | [ckpt](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_resnet152_ic15-6058a798.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_resnet152_ic15-6058a798-0d755205.mindir)   |
|     PSENet     |   ResNet-50   |    ImageNet    |     1     |       8        |      O2       |     185.16 s      |   280.21    |  228.40   |   76.55%   |    86.51%     |   81.23%    |      [yaml](pse_r50_icdar15.yaml)      | [ckpt](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_resnet50_ic15-7e36cab9.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_resnet50_ic15-7e36cab9-cfd2ee6c.mindir)    |
|     PSENet     |  MobileNetV3  |    ImageNet    |     8     |       8        |      O2       |     181.54 s      |   175.23    |  365.23   |   73.95%   |    67.78%     |   70.73%    |      [yaml](pse_mv3_icdar15.yaml)      |[ckpt](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_mobilenetv3_ic15-bf2c1907.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_mobilenetv3_ic15-bf2c1907-da7cfe09.mindir) |

### SCUT-CTW1500

| **model name** | **backbone**  | **pretrained** | **cards** | **batch size** | **jit level** | **graph compile** | **ms/step** | **img/s** | **recall** | **precision** | **f-score** |               **recipe**               |                                             **weight**                                            |
|:--------------:|:-------------:| :------------: |:---------:|:--------------:| :-----------: |:-----------------:|:-----------:|:---------:|:----------:|:-------------:|:-----------:|:--------------------------------------:|:-------------------------------------------------------------------------------------------------:|
|     PSENet     |  ResNet-152   |    ImageNet    |     8     |       8        |      O2       |     193.59 s      |   318.94    |  200.66   |   74.11%   |    73.45%     |   73.78%    |    [yaml](pse_r152_ctw1500.yaml)       | [ckpt](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_resnet152_ctw1500-58b1b1ff.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_resnet152_ctw1500-58b1b1ff-b95c7f85.mindir)  |

#### Notes：
- The training time of PSENet is highly affected by data processing and varies on different machines.
- The `input_shapes` to the exported MindIR models trained on ICDAR2015 are `(1,3,1472,2624)` for ResNet-152 backbone and `(1,3,736,1312)` for ResNet-50 or MobileNetV3 backbone.
- On the SCUT-CTW1500 dataset, the input_shape for exported MindIR in the link is `(1,3,1024,1024)`.

## References

<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Wang, Wenhai, et al. "Shape robust text detection with progressive scale expansion network." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.
