English | [中文](README_CN.md)

# DBNet and DBNet++

<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> DBNet: [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)  
> DBNet++: [Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion](https://arxiv.org/abs/2202.10304)

## 1. Introduction

### DBNet

DBNet is a segmentation-based scene text detection method. Segmentation-based methods are gaining popularity for scene
text detection purposes as they can more accurately describe scene text of various shapes, such as curved text.  
The drawback of current segmentation-based SOTA methods is the post-processing of binarization (conversion of
probability maps into text bounding boxes) which often requires a manually set threshold (reduces prediction accuracy)
and complex algorithms for grouping pixels (resulting in a considerable time cost during inference).  
To eliminate the problem described above, DBNet integrates an adaptive threshold called Differentiable Binarization(DB)
into the architecture. DB simplifies post-processing and enhances the performance of text detection.Moreover, it can be
removed in the inference stage without sacrificing performance.[[1](#references)]

<p align="center"><img alt="Figure 1. Overall DBNet architecture" src="https://user-images.githubusercontent.com/16683750/225589619-d50c506c-e903-4f59-a316-8b62586c73a9.png" width="800"/></p>
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

### DBNet++

DBNet++ is an extension of DBNet and thus replicates its architecture. The only difference is that instead of
concatenating extracted and scaled features from the backbone as DBNet did, DBNet++ uses an adaptive way to fuse those
features called Adaptive Scale Fusion (ASF) module (Figure 2). It improves the scale robustness of the network by
fusing features of different scales adaptively. By using ASF, DBNet++’s ability to detect text instances of diverse
scales is distinctly strengthened.[[2](#references)]

<p align="center"><img alt="Figure 2. Overall DBNet++ architecture" src="https://user-images.githubusercontent.com/16683750/236786997-13823b9c-ecaa-4bc5-8037-71299b3baffe.png" width="800"/></p>
<p align="center"><em>Figure 2. Overall DBNet++ architecture</em></p>

<p align="center"><img alt="Figure 3. Detailed architecture of the Adaptive Scale Fusion module" src="https://user-images.githubusercontent.com/16683750/236787093-c0c78d8f-e4f4-4c5e-8259-7120a14b0e31.png" width="700"/></p>
<p align="center"><em>Figure 3. Detailed architecture of the Adaptive Scale Fusion module</em></p>

ASF consists of two attention modules – stage-wise attention and spatial attention, where the latter is integrated in
the former as described in the Figure 3. The stage-wise attention module learns the weights of the feature maps of
different scales. While the spatial attention module learns the attention across the spatial dimensions. The
combination of these two modules leads to scale-robust feature fusion.  
DBNet++ performs better in detecting text instances of diverse scales, especially for large-scale text instances where
DBNet may generate inaccurate or discrete bounding boxes.

## 2. Results

DBNet and DBNet++ were trained on the ICDAR2015, MSRA-TD500, SCUT-CTW1500, Total-Text, and MLT2017 datasets. In addition, we conducted pre-training on the SynthText dataset and provided a url to download pretrained weights. All training results are as follows:


### ICDAR2015

<div align="center">

| **Model**           | **Context**    | **Backbone**  | **Pretrained** | **Recall** | **Precision** | **F-score** | **Train T.** | **Throughput** | **Recipe**                          | **Download**                                                                                                                                                                                              |
|---------------------|----------------|---------------|----------------|------------|---------------|-------------|--------------|----------------|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DBNet               | D910x1-MS2.0-G | MobileNetV3   | ImageNet       | 76.26%     | 78.22%        | 77.23%      | 10 s/epoch   | 100 img/s      | [yaml](db_mobilenetv3_icdar15.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_mobilenetv3-62c44539.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_mobilenetv3-62c44539-f14c6a13.mindir) |
| DBNet               | D910x1-MS2.0-G | ResNet-18     | ImageNet       | 80.12%     | 83.41%        | 81.73%      | 9.3 s/epoch  | 108 img/s      | [yaml](db_r18_icdar15.yaml)         | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18-0c0c4cfa.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18-0c0c4cfa-cf46eb8b.mindir)       |
| DBNet               | D910x1-MS2.0-G | ResNet-50     | ImageNet       | 83.53%     | 86.62%        | 85.05%      | 13.3 s/epoch | 75.2 img/s       | [yaml](db_r50_icdar15.yaml)         | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50-c3a4aa24.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50-c3a4aa24-fbf95c82.mindir)       |
|                     |                |               |                |            |               |             |              |                |                                     |                                                                                                                                                                                                           |
| DBNet++             | D910x1-MS2.0-G | ResNet-50     | [SynthText](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_synthtext-40655acb.ckpt)  | 85.70%     | 87.81%        | 86.74%      | 17.7 s/epoch | 56 img/s  | [yaml](db++_r50_icdar15.yaml)       | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnetpp_resnet50-068166c2.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnetpp_resnet50-068166c2-9934aff0.mindir)   |
</div>

### MSRA-TD500

<div align="center">

| **Model**         | **Context**    | **Backbone** | **Pretrained** | **Recall** | **Precision** | **F-score** | **Train T.** | **Throughput** | **Recipe**                  | **Download**                                                                                                                                                                                         |
|-------------------|----------------|--------------|----------------|------------|---------------|-------------|--------------|----------------|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DBNet            | D910x1-MS2.0-G | ResNet-18    | [SynthText](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18_synthtext-251ef3dd.ckpt)       | 79.55%     | 87.86%        | 83.50%      | 5.6 s/epoch  | 121.7 img/s      | [yaml](db_r18_td500.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18_td500-b5abff68.ckpt)  |
| DBNet           | D910x1-MS2.0-G | ResNet-50    | [SynthText](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_synthtext-40655acb.ckpt)       | 83.68%     | 87.59%        | 85.59%      | 9.6 s/epoch  | 71.2 img/s      | [yaml](db_r50_td500.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_td500-0d12b5e8.ckpt)  |
</div>

> MSRA-TD500 dataset has 300 training images and 200 testing images, reference paper [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947), we trained using an extra 400 traning images from HUST-TR400. You can down all [dataset](https://paddleocr.bj.bcebos.com/dataset/TD_TR.tar) for training.

### SCUT-CTW1500

<div align="center">

| **Model**         | **Context**    | **Backbone** | **Pretrained** | **Recall** | **Precision** | **F-score** | **Train T.** | **Throughput** | **Recipe**                  | **Download**                                                                                                                                                                                         |
|-------------------|----------------|--------------|----------------|------------|---------------|-------------|--------------|----------------|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DBNet            | D910x1-MS2.0-G | ResNet-18    | [SynthText](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18_synthtext-251ef3dd.ckpt)       | 85.68%     | 85.33%        | 85.50%      | 8.2 s/epoch  | 122.1 img/s      | [yaml](db_r18_ctw1500.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18_ctw1500-0864b040.ckpt)  |
| DBNet            | D910x1-MS2.0-G | ResNet-50    | [SynthText](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_synthtext-40655acb.ckpt)       | 86.72%     | 85.29%        | 86.00%      | 14.0 s/epoch  | 71.4 img/s      | [yaml](db_r50_ctw1500.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_ctw1500-f637e3d3.ckpt)  |
</div>

### Total-Text

<div align="center">

| **Model**         | **Context**    | **Backbone** | **Pretrained** | **Recall** | **Precision** | **F-score** | **Train T.** | **Throughput** | **Recipe**                  | **Download**                                                                                                                                                                                         |
|-----------------|----------------|--------------|----------------|------------|---------------|-------------|--------------|----------------|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DBNet            | D910x1-MS2.0-G | ResNet-18    | [SynthText](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18_synthtext-251ef3dd.ckpt)       | 83.66%     | 87.65%        | 85.61%      |    12.9 s/epoch   |  96.9 img/s        | [yaml](db_r18_totaltext.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18_totaltext-fb456ff4.ckpt)  |
| DBNet            | D910x1-MS2.0-G | ResNet-50    | [SynthText](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_synthtext-40655acb.ckpt)       | 84.79%     | 87.07%        | 85.91%      |    18.0 s/epoch  |   69.1 img/s        | [yaml](db_r50_totaltext.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_totaltext-76d6f421.ckpt)  |
</div>

### MLT2017

<div align="center">

| **Model**         | **Context**    | **Backbone** | **Pretrained** | **Recall** | **Precision** | **F-score** | **Train T.** | **Throughput** | **Recipe**                  | **Download**                                                                                                                                                                                         |
|-------------------|----------------|--------------|----------------|------------|---------------|-------------|--------------|----------------|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DBNet            | D910x8-MS2.0-G | ResNet-18    | [SynthText](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18_synthtext-251ef3dd.ckpt)       | 72.55%     | 83.23%        | 77.52%       |  20.9 s/epoch  |   43.1 img/s      | [yaml](db_r18_mlt2017.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18_mlt2017-5af33809.ckpt)  |
| DBNet           | D910x8-MS2.0-G | ResNet-50    | [SynthText](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_synthtext-40655acb.ckpt)        | 74.88%     | 83.77%        | 79.08%       |  23.6 s/epoch  |   38.2 img/s      | [yaml](db_r50_mlt2017.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_mlt2017-3bd6e569.ckpt)  |
</div>

### SynthText

<div align="center">

| **Model**         | **Context**    | **Backbone** | **Pretrained** |  **Train Loss**|  **Train T.** | **Throughput** | **Recipe**                  | **Download**                 |
|-------------------|----------------|--------------|----------------|-------------|------------|---------------|-------------|--------------|
| DBNet        | D910x1-MS2.0-G | ResNet-18    | ImageNet       |       2.41    |7075 s/epoch   | 121.37 img/s     | [yaml](db_r18_synthtext.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18_synthtext-251ef3dd.ckpt)  |
| DBNet        | D910x1-MS2.0-G | ResNet-50    | ImageNet       |       2.25    |10470 s/epoch  | 82.02 img/s      | [yaml](db_r50_synthtext.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_synthtext-40655acb.ckpt)  |
</div>


#### Notes
- Context: Training context denoted as {device}x{pieces}-{MS version}{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Note that the training time of DBNet is highly affected by data processing and varies on different machines. 



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
│   ├── images
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   └── ...
│   └── test_det_gt.txt
└── train
    ├── images
    │   ├── img_1.jpg
    │   ├── img_2.jpg
    │   └── ....jpg
    └── train_det_gt.txt
```

#### 3.2.2 MSRA-TD500 dataset

Please download [MSRA-TD500](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500)) dataset，and convert the labels to the desired format referring to [dataset_converters](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters/README.md).

The prepared dataset file struture should be: 

```txt
MSRA-TD500
 ├── test
 │   ├── IMG_0059.gt 
 │   ├── IMG_0059.JPG
 │   ├── IMG_0080.gt
 │   ├── IMG_0080.JPG
 │   ├── ...
 │   ├── train_det_gt.txt
 ├── train
 │   ├── IMG_0030.gt 
 │   ├── IMG_0030.JPG
 │   ├── IMG_0063.gt
 │   ├── IMG_0063.JPG
 │   ├── ...
 │   ├── test_det_gt.txt
```

#### 3.2.3 SCUT-CTW1500 dataset

Please download [SCUT-CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector) dataset，and convert the labels to the desired format referring to [dataset_converters](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters/README.md).

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
 ├── train_det_gt.txt
```

#### 3.2.4 Total-Text dataset

Please download [Total-Text](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Dataset) dataset，and convert the labels to the desired format referring to [dataset_converters](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters/README.md).

The prepared dataset file struture should be: 


```txt
totaltext
 ├── Images
 │   ├── Train
 │   │   ├── img1001.jpg
 │   │   ├── img1002.jpg
 │   │   ├── ...
 │   ├── Test
 │   │   ├── img1.jpg
 │   │   ├── img2.jpg
 │   │   ├── ...
 ├── test_det_gt.txt
 ├── train_det_gt.txt
```

#### 3.2.5 MLT2017 dataset

The MLT2017 dataset is a multilingual text detection and recognition dataset that includes nine languages: Chinese, Japanese, Korean, English, French, Arabic, Italian, German, and Hindi. Please download [MLT2017](https://rrc.cvc.uab.es/?ch=8&com=downloads) and extract the dataset. Then convert the .gif format images in the data to .jpg or .png format, and convert the labels to the desired format referring to [dataset_converters](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters/README.md).

The prepared dataset file struture should be: 

```txt
MLT_2017
 ├── train
 │   ├── img_1.png
 │   ├── img_2.png
 │   ├── img_3.jpg
 │   ├── img_4.jpg
 │   ├── ...
 ├── validation
 │   ├── img_1.jpg
 │   ├── img_2.jpg
 │   ├── ...
 ├── train_det_gt.txt
 ├── validation_det_gt.txt
```

> If users want to use their own dataset for training, please convert the labels to the desired format referring to [dataset_converters](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters/README.md). Then configure the yaml file, and use a single or multiple devices to run train.py for training. For detailed information, please refer to the following tutorials.

#### 3.2.6 SynthText dataset

Please download [SynthText](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c) dataset，The directory structure of the extracted data should be as follows:

``` text
.
├── SynthText
│   ├── 1
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   └── ...
│   ├── 2
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   └── ...
│   ├── ...
│   ├── 200
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   └── ...
│   └── gt.mat

```

### 3.3 Update yaml config file

Update `configs/det/dbnet/db_r50_icdar15.yaml` configuration file with data paths,
specifically the following parts. The `dataset_root` will be concatenated with `dataset_root` and `label_file` respectively to be the complete dataset directory and label file path.

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



DBNet consists of 3 parts: `backbone`, `neck`, and `head`. Specifically:

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

### 3.4 Training

* Standalone training

Please set `distribute` in yaml config file to be False.

```shell
python tools/train.py -c=configs/det/dbnet/db_r50_icdar15.yaml
```

* Distributed training

Please set `distribute` in yaml config file to be True.

```shell
# n is the number of GPUs/NPUs
mpirun --allow-run-as-root -n 2 python tools/train.py --config configs/det/dbnet/db_r50_icdar15.yaml
```
 
The training result (including checkpoints, per-epoch performance and curves) will be saved in the directory parsed by the arg `ckpt_save_dir` in yaml config file. The default directory is `./tmp_det`.


### 3.5 Evaluation

To evaluate the accuracy of the trained model, you can use `eval.py`. Please set the checkpoint path to the arg `ckpt_load_path` in the `eval` section of yaml config file, set `distribute` to be False, and then run:

```shell
python tools/eval.py -c=configs/det/dbnet/db_r50_icdar15.yaml
```

## References

<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Minghui Liao, Zhaoyi Wan, Cong Yao, Kai Chen, Xiang Bai. Real-time Scene Text Detection with Differentiable
Binarization. arXiv:1911.08947, 2019

[2] Minghui Liao, Zhisheng Zou, Zhaoyi Wan, Cong Yao, Xiang Bai. Real-Time Scene Text Detection with Differentiable
Binarization and Adaptive Scale Fusion. arXiv:2202.10304, 2022
