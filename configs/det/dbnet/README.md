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


### Requirements

| mindspore | ascend driver | firmware     | cann toolkit/kernel
|:----------|:---           | :--          |:--
| 2.3.1     | 24.1.RC2      | 7.3.0.1.231  | 8.0.RC2.beta1

## 2. General purpose models

Here we present general purpose models that were trained on wide variety of tasks (real-world photos, street views, documents, etc.) and challenges (straight texts, curved texts, long text lines, etc.) with two primary languages: Chinese and English. These models can be used right off-the-shelf in your applications or for initialization of your models.

The models were trained on 12 public datasets (CTW, LSVT, RCTW-17, TextOCR, etc.) that contain wide range of images. The training set has 153,511 images and the validation set has 9,786 images.<br/>
The test set consists of 598 images manually selected from the above-mentioned datasets.

Experiments are tested on ascend 910 with mindspore 2.3.1 graph mode

<div align="center">

| **model name** | **backbone** | **cards** | **batch size** |   **languages**   | **jit level** | **graph compile** | **ms/step** | **img/s**  | **f-score** | **recipe**     | **download** |
|----------------|-----------|----------------|--------------|:-----------------:|-------------|:-----------------:|-------------|------------|-------------|-----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| DBNet          | ResNet-50    | 8         | 10             | Chinese + English | O2|     107.91 s      | 312.48      | 256        | 83.41%      | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_ch_en_general-a5dbb141.ckpt)   | [mindir](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_ch_en_general-a5dbb141-912f0a90.mindir)   |
| DBNet++        | ResNet-50    | 4         | 32             | Chinese + English | O2|     182.94 s      | 1230.76     | 104|  84.30%     | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnetpp_resnet50_ch_en_general-884ba5b9.ckpt) | [mindir](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnetpp_resnet50_ch_en_general-884ba5b9-b3f52398.mindir) |
</div>

> The input_shape for exported DBNet MindIR and DBNet++ MindIR in the links are `(1,3,736,1280)` and `(1,3,1152,2048)`, respectively.


## 3. Results

DBNet and DBNet++ were trained on the ICDAR2015, MSRA-TD500, SCUT-CTW1500, Total-Text, and MLT2017 datasets. In addition, we conducted pre-training on the SynthText dataset and provided a URL to download pretrained weights. All training results are as follows:

<details>
   <summary>Experiments are tested on ascend 910 with mindspore 2.3.1 graph mode</summary>

   ### ICDAR2015

   <div align="center">

   | **model name** | **backbone** | **cards** | **batch size** | **pretrained** | **ms/step** | img/s                                   | **recall** | **precision** | **f-score** | **recipe**                             | **download**                                                                                                                                                                                                    |
   |----------------|-----------|----------------|--------------|----------------|------------|---------------|-------------|-------------|-----------------------------------------|----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
   | DBNet          | MobileNetV3  | 1         | 10             | ImageNet       | 100         | 100                                     | 76.31%     | 78.27%        | 77.28%      | [yaml](db_mobilenetv3_icdar15.yaml)    | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_mobilenetv3-62c44539.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_mobilenetv3-62c44539-f14c6a13.mindir)       |
   | DBNet          | MobileNetV3  | 8         | 8              | ImageNet       | 66.64       | 960                                     | 76.22%     | 77.98%        | 77.09%      | [yaml](db_mobilenetv3_icdar15_8p.yaml) | Coming soon                                                                                                                                                                                                     |
   | DBNet          | ResNet-18    | 1         | 20             | ImageNet       | 185.19      | 108                                     | 80.12%     | 83.41%        | 81.73%      | [yaml](db_r18_icdar15.yaml)            | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18-0c0c4cfa.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18-0c0c4cfa-cf46eb8b.mindir)             |
   | DBNet          | ResNet-50    | 1         | 10             | ImageNet       | 132.98      | 75.2                                    | 83.53%     | 86.62%        | 85.05%      | [yaml](db_r50_icdar15.yaml)            | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50-c3a4aa24.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50-c3a4aa24-fbf95c82.mindir)             |
   | DBNet          | ResNet-50    | 8         | 10             | ImageNet       | 183.92      | 435                                     | 82.62%     | 88.54%        | 85.48%      | [yaml](db_r50_icdar15_8p.yaml)         | Coming soon                                                                                                                                                                                                     |
   |                |           |                |              |                |            |               |             |             |                                         |                                        |
   | DBNet++        | ResNet-50    | 1         | 32             | SynthText      | 409.21      | 78,.2| 86.81%     | 86.85%        | 86.86%      |  [yaml](dbpp_r50_icdar15_910.yaml)     | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnetpp_resnet50_910-35dc71f2.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnetpp_resnet50_910-35dc71f2-e61a9c37.mindir) |
   </div>

   > The input_shape for exported DBNet MindIR and DBNet++ MindIR in the links are `(1,3,736,1280)` and `(1,3,1152,2048)`, respectively.

   ### MSRA-TD500

   <div align="center">

   | **model name** | **backbone** | **cards** | **batch size** | **pretrained** | **ms/step** | **img/s**   | **recall** | **precision** | **f-score** | **recipe**                | **download**                                                                                    |
   |----------------|---------|----------------|--------------|----------------|-------------|-------------|------------|-------------|------------------|---------------------------|-------------------------------------------------------------------------------------------------|
   | DBNet          | ResNet-18    | 1       | 20             | SynthText      | 163.34      | 121.7       | 79.90%     | 88.07%        | 83.78%      | [yaml](db_r18_td500.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18_td500-b5abff68.ckpt) |
   | DBNet          | ResNet-50    | 1       | 20             | SynthText      | 280.90      | 71.2|  84.02%    | 87.48%     | 85.71%      | [yaml](db_r50_td500.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_td500-0d12b5e8.ckpt) |
   </div>

   > MSRA-TD500 dataset has 300 training images and 200 testing images, reference paper [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947), we trained using an extra 400 traning images from HUST-TR400. You can down all [dataset](https://paddleocr.bj.bcebos.com/dataset/TD_TR.tar) for training.

   ### SCUT-CTW1500

   <div align="center">

   | **model name** | **backbone** | **cards** | **batch size** | **pretrained** | **ms/step** | **img/s**   | **recall** | **precision** | **f-score** | **recipe**                  | **download**                                                                                      |
   |----------------|---------|----------------|--------------|----------------|------------|-------------|------------|-------------|-------------|-----------------------------|---------------------------------------------------------------------------------------------------|
   | DBNet          | ResNet-18    | 1       | 20             | SynthText      | 163.80      | 122.1       | 85.68%     | 85.33%        | 85.50%      | [yaml](db_r18_ctw1500.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18_ctw1500-0864b040.ckpt) |
   | DBNet          | ResNet-50    | 1       | 20             | SynthText      | 180.11      | 71.4|  87.83%    | 84.71%     | 86.25%      | [yaml](db_r50_ctw1500.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_ctw1500-f637e3d3.ckpt) |
   </div>

   ### Total-Text

   <div align="center">

   | **model name** | **backbone** | **cards** | **batch size** | **pretrained** | **ms/step** | **img/s**   | **recall** | **precision** | **f-score** | **recipe**                    | **download**                                                                                        |
   |----------------|---------|----------------|--------------|----------------|------------|-------------|------------|-------------|-------------|-------------------------------|-----------------------------------------------------------------------------------------------------|
   | DBNet          | ResNet-18    | 1       | 20             | SynthText      | 206.40      | 96.9        | 83.66%     | 87.61%        | 85.59%      | [yaml](db_r18_totaltext.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18_totaltext-fb456ff4.ckpt) |
   | DBNet          | ResNet-50    | 1       | 20             | SynthText      | 289.44      | 69.1|  84.79%    | 87.07%     | 85.91%      | [yaml](db_r50_totaltext.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_totaltext-76d6f421.ckpt) |
   </div>

   ### MLT2017

   <div align="center">

   | **model name** | **backbone** | **cards** | **batch size** | **pretrained** | **ms/step** | **img/s**    | **recall** | **precision** | **f-score** | **recipe**                  | **download**                                                                                      |
   |----------------|---------|----------------|--------------|----------------|------------|--------------|------------|-------------|-------------|-----------------------------|---------------------------------------------------------------------------------------------------|
   | DBNet          | ResNet-18    | 8       | 20             | SynthText      | 464.00      | 344.8        | 73.62%     | 83.93%        | 78.44%      | [yaml](db_r18_mlt2017.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18_mlt2017-5af33809.ckpt) |
   | DBNet          | ResNet-50    | 8       | 20             | SynthText      | 523.60      | 305.6|  76.04%    | 84.51%     | 80.05%      | [yaml](db_r50_mlt2017.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_mlt2017-3bd6e569.ckpt) |
   </div>

   ### SynthText

   <div align="center">

   | **model name** | **backbone** | **cards** | **batch size** | **pretrained** | **ms/step** | **img/s**  | **train loss** | **recipe**                    | **download**                                                                                        |
   |----------------|---------|----------------|--------------|----------------|----------------|------------|----------------|-------------------------------|-----------------------------------------------------------------------------------------------------|
   | DBNet          | ResNet-18    | 1       | 16             | ImageNet       | 131.83      | 121.37     | 2.41           | [yaml](db_r18_synthtext.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18_synthtext-251ef3dd.ckpt) |
   | DBNet          | ResNet-50    | 1       | 16             | ImageNet       | 195.07      | 82.02|  2.25          | [yaml](db_r50_synthtext.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50_synthtext-40655acb.ckpt) |
   </div>

</details>

<details>
   <summary>Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode</summary>

   ### ICDAR2015

   <div align="center">

   | **model name** | **backbone** | **cards** | **batch size** | **pretrained** | **ms/step** | **img/s** | **recall** | **precision** | **f-score** | **recipe**                             | **download**                                                                                               |
   |----------------|-------|----------------|--------------|----------------|------------|-----------|------------|-------------|-------------|----------------------------------------|------------------------------------------------------------------------------------------------------------|
   | DBNet          | MobileNetV3  | 1     | 10             | ImageNet       | 65.69       | 152.23    | 74.68%     | 79.38%        | 76.95%      | [yaml](db_mobilenetv3_icdar15.yaml)    | [ckpt](https://download-mindspore.osinfra.cn/toolkits/mindocr/dbnet/dbnet_mobilenetv3-e72f9b8b-910v2.ckpt) |
   | DBNet          | MobileNetV3  | 8     | 8              | ImageNet       | 54.46       | 1175.12   | 76.27%     | 76.06%     | 76.17%      | [yaml](db_mobilenetv3_icdar15_8p.yaml) | [ckpt](https://download-mindspore.osinfra.cn/toolkits/mindocr/dbnet/dbnet_mobilenetv3-7e89e1df-910v2.ckpt) |
   | DBNet          | ResNet-50    | 1     | 10             | ImageNet       | 155.62      | 64.25     | 84.50%     | 85.36%     | 84.93%      | [yaml](db_r50_icdar15.yaml)            | [ckpt](https://download-mindspore.osinfra.cn/toolkits/mindocr/dbnet/dbnet_resnet50-48153c3b-910v2.ckpt)    |
   | DBNet          | ResNet-50    | 8     | 10             | ImageNet       | 159.22      | 502.4     |  81.15%    | 87.63%     | 84.26%      | [yaml](db_r50_icdar15_8p.yaml)         | [ckpt](https://download-mindspore.osinfra.cn/toolkits/mindocr/dbnet/dbnet_resnet50-e10bad35-910v2.ckpt)    |

   </div>

   > The input_shape for exported DBNet MindIR and DBNet++ MindIR in the links are `(1,3,736,1280)` and `(1,3,1152,2048)`, respectively.

</details>


#### Notes
- Note that the training time of DBNet is highly affected by data processing and varies on different machines.



## 4. Quick Start

### 4.1 Installation

Please refer to the [installation instruction](https://github.com/mindspore-lab/mindocr#installation) in MindOCR.

### 4.2 Dataset preparation

#### 4.2.1 ICDAR2015 dataset

Please download [ICDAR2015](https://rrc.cvc.uab.es/?ch=4&com=downloads) dataset, and convert the labels to the desired format referring to [dataset_converters](../../../tools/dataset_converters/README.md).

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

#### 4.2.2 MSRA-TD500 dataset

Please download [MSRA-TD500](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500)) dataset，and convert the labels to the desired format referring to [dataset_converters](../../../tools/dataset_converters/README.md).

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

#### 4.2.3 SCUT-CTW1500 dataset

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

#### 4.2.4 Total-Text dataset

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

#### 4.2.5 MLT2017 dataset

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

#### 4.2.6 SynthText dataset

Please download [SynthText](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c) dataset and process it as described in [dataset_converters](../../../tools/dataset_converters/README.md)

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

> :warning: Additionally, It is strongly recommended to pre-process the `SynthText` dataset before using it as it contains some faulty data:
> ```shell
> python tools/dataset_converters/convert.py --dataset_name=synthtext --task=det --label_dir=/path-to-data-dir/SynthText/gt.mat --output_path=/path-to-data-dir/SynthText/gt_processed.mat
> ```
> This operation will generate a filtered output in the same format as the original `SynthText`.

### 4.3 Update yaml config file

Update `configs/det/dbnet/db_r50_icdar15.yaml` configuration file with data paths,
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

### 4.4 Training

* Standalone training

Please set `distribute` in yaml config file to be False.

```shell
python tools/train.py -c=configs/det/dbnet/db_r50_icdar15.yaml
```

* Distributed training

Please set `distribute` in yaml config file to be True.

```shell
# n is the number of NPUs
mpirun --allow-run-as-root -n 2 python tools/train.py --config configs/det/dbnet/db_r50_icdar15.yaml
```

The training result (including checkpoints, per-epoch performance and curves) will be saved in the directory parsed by the arg `ckpt_save_dir` in yaml config file. The default directory is `./tmp_det`.


### 4.5 Evaluation

To evaluate the accuracy of the trained model, you can use `eval.py`. Please set the checkpoint path to the arg `ckpt_load_path` in the `eval` section of yaml config file, set `distribute` to be False, and then run:

```shell
python tools/eval.py -c=configs/det/dbnet/db_r50_icdar15.yaml
```

## 5. MindSpore Lite Inference

Please refer to the tutorial [MindOCR Inference](../../../docs/en/inference/inference_tutorial.md) for model inference based on MindSpot Lite on Ascend 310, including the following steps:

- Model Export

Please [download](#3-results) the exported MindIR file first, or refer to the [Model Export](../../../docs/en/inference/convert_tutorial.md#1-model-export) tutorial and use the following command to export the trained ckpt model to  MindIR file:

```shell
python tools/export.py --model_name_or_config dbnet_resnet50 --data_shape 736 1280 --local_ckpt_path /path/to/local_ckpt.ckpt
# or
python tools/export.py --model_name_or_config configs/det/dbnet/db_r50_icdar15.yaml --data_shape 736 1280 --local_ckpt_path /path/to/local_ckpt.ckpt
```

The `data_shape` is the model input shape of height and width for MindIR file. The shape value of MindIR in the download link can be found in [ICDAR2015 Notes](#ICDAR2015).

- Environment Installation

Please refer to [Environment Installation](../../../docs/en/inference/environment.md) tutorial to configure the MindSpore Lite inference environment.

- Model Conversion

Please refer to [Model Conversion](../../../docs/en/inference/convert_tutorial.md#2-mindspore-lite-mindir-convert),
and use the `converter_lite` tool for offline conversion of the MindIR file.

- Inference

Assuming that you obtain output.mindir after model conversion, go to the `deploy/py_infer` directory, and use the following command for inference:

```shell
python infer.py \
    --input_images_dir=/your_path_to/test_images \
    --det_model_path=your_path_to/output.mindir \
    --det_model_name_or_config=../../configs/det/dbnet/db_r50_icdar15.yaml \
    --res_save_dir=results_dir
```

## References

<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Minghui Liao, Zhaoyi Wan, Cong Yao, Kai Chen, Xiang Bai. Real-time Scene Text Detection with Differentiable
Binarization. arXiv:1911.08947, 2019

[2] Minghui Liao, Zhisheng Zou, Zhaoyi Wan, Cong Yao, Xiang Bai. Real-Time Scene Text Detection with Differentiable
Binarization and Adaptive Scale Fusion. arXiv:2202.10304, 2022
