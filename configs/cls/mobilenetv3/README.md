English | [中文](README_CN.md)

# MobileNetV3 for text direction classification

## Introduction

### MobileNetV3: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

MobileNetV3[[1](#references)] was published in 2019, which combines the deep separable convolution of V1, the Inverted Residuals and Linear Bottleneck of V2, and the SE (Squeeze and Excitation) module to search the configuration and parameters of the network using NAS (Neural Architecture Search). MobileNetV3 first uses MnasNet to perform a coarse structure search, and then uses reinforcement learning to select the optimal configuration from a set of discrete choices. Besides, MobileNetV3 fine-tunes the architecture using NetAdapt. Overall, MobileNetV3 is a lightweight network having good performance in classification, detection and segmentation tasks.

<p align="center">
  <img src="https://user-images.githubusercontent.com/53842165/210044297-d658ca54-e6ff-4c0f-8080-88072814d8e6.png" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of MobileNetV3 [<a href="#references">1</a>] </em>
</p>


### Text direction classifier

The text directions in some images are revered, so that the text cannot be regconized correctly. Therefore. we use a text direction classifier to classify and rectify the text direction. The MobileNetV3 paper releases two versions of MobileNetV3: *MobileNetV3-Large* and *MobileNetV3-Small*. Taking the tradeoff between efficiency and accuracy, we adopt the *MobileNetV3-Small* as the text direction classifier.

Currently we support the 0 and 180 degree classification. You can update the params `label_list` and `num_classes` in yaml config file to train your own classifier. The 0 and 180 degree data samples are shown below.

<div align="center">

| **0 degree**         | **180 degree**    |
|-------------------|----------------|
|<img src="https://github.com/HaoyangLee/mindocr/assets/20376974/7dd4432f-775c-4a04-b9e7-0f52f68c70ee" width=200 /> | <img src="https://github.com/HaoyangLee/mindocr/assets/20376974/cfe298cd-08be-4866-b650-5eae560d59fa" width=200 /> |
|<img src="https://github.com/HaoyangLee/mindocr/assets/20376974/666da152-2e9b-48ae-b1a7-e190deef34d6" width=200 /> | <img src="https://github.com/HaoyangLee/mindocr/assets/20376974/802617fe-910e-451d-b202-e05da47b1196" width=200 /> |

</div>


## Results

| mindspore |  ascend driver  |   firmware   | cann toolkit/kernel |
|:---------:|:---------------:|:------------:|:-------------------:|
|   2.3.1   |    24.1.RC2     | 7.3.0.1.231  |    8.0.RC2.beta1    |

MobileNetV3 is pretrained on ImageNet. For text direction classification task, we further train MobileNetV3 on RCTW17, MTWI and LSVT datasets.

Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode
<div align="center">

| **model name** | **cards** | **batch size** | **img/s** | **accuracy** | **config**  | **weight**                                                                            |
|----------------|-----------|----------------|-----------|--------------|-----------------------------------------------------|------------------------------------------------|
| MobileNetV3    | 4         | 256            | 5923.5    | 94.59%       | [yaml](cls_mv3.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/cls/cls_mobilenetv3-92db9c58.ckpt) |
</div>




## Quick Start

### Installation

Please refer to the [installation instruction](https://github.com/mindspore-lab/mindocr#installation) in MindOCR.

### Dataset preparation

Please download [RCTW17](https://rctw.vlrlab.net/dataset), [MTWI](https://tianchi.aliyun.com/competition/entrance/231684/introduction), and [LSVT](https://rrc.cvc.uab.es/?ch=16&com=introduction) datasets, and then process the images and labels in desired format referring to [dataset_converters](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters/README.md).

The prepared dataset file struture is suggested to be as follows.

``` text
.
├── all_images
│   ├── img_1.jpg
│   ├── img_2.jpg
│   └── ...
├── eval_cls_gt.txt
├── train_cls_gt.txt
└── val_cls_gt.txt
```

> If you want to use your own dataset for training, please convert the images and labels to the desired format referring to [dataset_converters](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters/README.md).


### Update yaml config file

Update the dataset directories in yaml config file. The `dataset_root` will be concatenated with `data_dir` and `label_file` respectively to be the complete image directory and label file path.

```yaml
...
train:
  ckpt_save_dir: './tmp_cls'
  dataset:
    type: RecDataset
    dataset_root: dir/to/dataset          <--- Update
    data_dir: all_images                  <--- Update
    label_file: train_cls_gt.txt          <--- Update
...
eval:
  dataset:
    type: RecDataset
    dataset_root: dir/to/dataset          <--- Update
    data_dir: all_images                  <--- Update
    label_file: val_cls_gt.txt            <--- Update
...
```

> Optionally, change `num_workers` according to the cores of CPU.



MobileNetV3 for text dierection classification consists of 2 parts: `backbone` and `head`.

```yaml
model:
  type: cls
  transform: null
  backbone:
    name: cls_mobilenet_v3_small_100
    pretrained: True
  head:
    name: MobileNetV3Head
    out_channels: 1024  # arch=small 1024, arch=large 1280
    num_classes: *num_classes  # 2 or 4
```


### Evaluation

Please set the checkpoint path to the arg `ckpt_load_path` in the `eval` section of yaml config file, set `distribute` to be `False`, and then run:

```shell
python tools/eval.py -c configs/cls/mobilenetv3/cls_mv3.yaml
```

## References

<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Howard A, Sandler M, Chu G, et al. Searching for mobilenetv3[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2019: 1314-1324.
