English | [中文](README_CN.md)

# FCENet

<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> FCENet: [Fourier Contour Embedding for Arbitrary-Shaped Text Detection](https://arxiv.org/pdf/2104.10442.pdf)

## 1. Introduction

### FCENet

FCENet is a segmentation-based text detection algorithm. In the text detection scene, algorithms based on segmentation are becoming increasingly popular as they can accurately describe arbitrary shapes of text, including highly-curved text. One of the key highlights of FCENet is its excellent performance on arbitrary-shaped text scenes, which is achieved through deformable convolutions [1] and Fourier transform techniques. Additionally, FCENet possesses advantages of simple post-processing and high generalization, allowing it to achieve good results even with limited training data.

#### Deformable Convolution

The idea of deformable convolution is very simple, which is to change the fixed shape of the convolution kernel into a variable one. Based on the position of the original convolution, deformable convolution will generate a random position shift, as shown in the following figure:

<p align="center"><img alt="Figure 1" src="https://github.com/colawyee/mindocr/assets/15730439/20cdb21d-f0b4-4fdf-a8dd-5833d84648ba" width="600"/></p>
<p align="center"><em>Figure 1. Deformable Convolution</em></p>

Figure (a) is the original convolutional kernel, Figure (b) is a deformable convolutional kernel that generates random directional position shifts, and Figure (c) and (d) are two special cases of Figure (b). It can be seen that the advantage of this is that it can improve the Geometric transformation ability of the convolution kernel, so that it is not limited to the shape of the original convolution kernel rectangle, but can support more abundant irregular shapes. Deformable convolution performs better in extracting irregular shape features [1] (# References)] and is more suitable for text recognition scenarios in natural scenes.

#### Fourier Contour

Fourier contour is a curve fitting method based on Fourier transform. As the number of Fourier degree k increases, more high-frequency signals will be introduced, and the contour description will be more accurate. The following figure shows the ability to describe irregular curves under different Fourier degree:

<p align="center"><img width="445" alt="Image" src="https://github.com/colawyee/mindocr/assets/15730439/ae507f2f-ea4d-4787-90ea-4c59c634567a"></p>
<p align="center"><em>Figure 2. Fourier contour fitting with progressive approximation</em></p>

It can be seen that as the Fourier degree k increases, the curves it can depict can become very complicated.

#### Fourier Contour Embedding

Fourier Contour Encoding is a method proposed in the paper "Fourier Contour Embedding for Arbitrary Shaped Text Detection" to convert the closed text contour curve into a vector. It is also a fundamental ability required for FCENet algorithm to encode contour lines. This method samples points at equal intervals on the contour line, and then converts the sequence of sampled points into Fourier feature vectors. It is worth noting that even for the same contour line, if the sampling points are different, the corresponding generated Fourier feature vectors are not the same. So when sampling, it is necessary to limit the starting point, uniform speed, and sampling direction to ensure the uniqueness of Fourier feature vectors generated for the same contour line.

#### The FCENet Framework

<p align="center"><img width="800" alt="Image" src="https://github.com/colawyee/mindocr/assets/15730439/347aed5a-454c-4cfc-9577-fee163239626"></p>
<p align="center"><em>Figure 3. FCENet framework</em></p>

Like most OCR algorithms, the structure of FCENet can be roughly divided into three parts: backbone, neck, and head. The backbone uses a deformable convolutional version of Resnet50 for feature extraction; The neck section adopts a feature pyramid [2] (# References), which is a set of convolutional kernels of different sizes, suitable for extracting features of different sizes from the original image, thereby improving the accuracy of object detection. It suits scenes that there are a few text boxes of different sizes in one image; The head part has two branches, one is the classification branch. The classification branch predicts the heat maps of both text regions and text center regions, which are pixel-wise multiplied, resulting in the the classification score map. The loss of classification branch is calculated by the cross entropy between prediction heat maps and ground truth. The regression branch predicts the Fourier signature vectors, which are used to reconstruct text contours via the Inverse Fourier transformation (IFT). Calculate the smooth-l1 loss of the reconstructed text contour and the ground truth contour in the image space as the loss value of the regression branch.

## 2. Results

The FCENet in MindOCR is trained on ICDAR 2015 dataset. The training results are as follows:

### ICDAR2015

<div align="center">

| **Model**              | **Context**       | **Backbone**      | **Pretrained** | **Recall** | **Precision** | **F-score** | **Train T.**     | **Throughput**   | **Recipe**                            | **Download**                                                                                                                                                                                                |
|---------------------|----------------|---------------|------------|------------|---------------|-------------|--------------|-----------|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FCENet               | D910x4-MS2.0-F | ResNet50   | ImageNet       | 81.5%     | 86.9%        | 84.1%      | 33 s/epoch   | 7 img/s      | [yaml](fce_icdar15.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/fcenet/) \| [mindir](https://download.mindspore.cn/toolkits/mindocr.mindir) |

</div>

#### Notes
- Context: Training context denoted as {device}x{pieces}-{MS version}{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Note that the training time of FCENet is highly affected by data processing and varies on different machines.

## 3. Quick Start

### 3.1 Installation

Please refer to the [installation instruction](https://github.com/mindspore-lab/mindocr#installation) in MindOCR.

### 3.2 Dataset preparation

#### 3.2.1 ICDAR2015 dataset

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

### 3.3 Update yaml config file

Update `configs/det/fcenet/fce_icdar15.yaml` configuration file with data paths,
specifically the following parts. The `dataset_root` will be concatenated with `data_dir` and `label_file` respectively to be the complete dataset directory and label file path.

```yaml
...
train:
  ckpt_save_dir: './tmp_det_fcenet'
  dataset_sink_mode: False
  ema: True
  dataset:
    type: DetDataset
    dataset_root: dir/to/dataset          <--- Update
    data_dir: icda/ch4_training_images    <--- Update
    label_file: icda/train_det_gt.txt     <--- Update
...
eval:
  ckpt_load_path: '/best.ckpt'            <--- Update
  dataset_sink_mode: False
  dataset:
    type: DetDataset
    dataset_root: dir/to/dataset          <--- Update
    data_dir: icda/ch4_test_images        <--- Update
    label_file: icda/test_det_gt.txt      <--- Update
...
```

> Optionally, change `num_workers` according to the cores of CPU.

FCENet consists of 3 parts: `backbone`, `neck`, and `head`. Specifically:

```yaml
model:
  resume: False
  type: det
  transform: null
  backbone:
    name: det_resnet50  # Only ResNet50 is supported at the moment
    pretrained: True    # Whether to use weights pretrained on ImageNet
  neck:
    name: FCEFPN        # FPN part of the FCENet
    out_channel: 256
  head:
    name: FCEHead
    scales: [ 8, 16, 32 ]
    alpha: 1.2
    beta: 1.0
    fourier_degree: 5
    num_sample: 50
```

### 3.4 Training

* Standalone training

Please set `distribute` in yaml config file to be False.

```shell
python tools/train.py -c=configs/det/fcenet/fce_icdar15.yaml
```

* Distributed training

Please set `distribute` in yaml config file to be True.

```shell
# n is the number of GPUs/NPUs
mpirun --allow-run-as-root -n 2 python tools/train.py --config configs/det/fcenet/fce_icdar15.yaml
```

The training result (including checkpoints, per-epoch performance and curves) will be saved in the directory parsed by the arg `ckpt_save_dir` in yaml config file. The default directory is `./tmp_det`.


### 3.5 Evaluation

To evaluate the accuracy of the trained model, you can use `eval.py`. Please set the checkpoint path to the arg `ckpt_load_path` in the `eval` section of yaml config file, set `distribute` to be False, and then run:

```shell
python tools/eval.py -c=configs/det/fcenet/fce_icdar15.yaml
```

### 3.6 MindSpore Lite Inference

Please refer to the tutorial [MindOCR Inference](../../../docs/en/inference/inference_tutorial_en.md) for model inference based on MindSpot Lite on Ascend 310, including the following steps:

- Model Export

Please [download](#2-results) the exported MindIR file first, or refer to the [Model Export](../../README.md) tutorial and use the following command to export the trained ckpt model to  MindIR file:

```shell
python tools/export.py --model_name fcenet_resnet50 --data_shape 736 1280 --local_ckpt_path /path/to/local_ckpt.ckpt
# or
python tools/export.py --model_name configs/det/fcenet/fce_icdar15.yaml --data_shape 736 1280 --local_ckpt_path /path/to/local_ckpt.ckpt
```

The `data_shape` is the model input shape of height and width for MindIR file. The shape value of MindIR in the download link can be found in [ICDAR2015 Notes](#ICDAR2015).

- Environment Installation

Please refer to [Environment Installation](../../../docs/en/inference/environment_en.md#2-mindspore-lite-inference) tutorial to configure the MindSpore Lite inference environment.

- Model Conversion

Please refer to [Model Conversion](../../../docs/en/inference/convert_tutorial_en.md#1-mindocr-models),
and use the `converter_lite` tool for offline conversion of the MindIR file, where the `input_shape` in `configFile` needs to be filled in with the value from MindIR export,
as mentioned above (1, 3, 736, 1280), and the format is NCHW.

- Inference

Assuming that you obtain output.mindir after model conversion, go to the `deploy/py_infer` directory, and use the following command for inference:

```shell
python infer.py \
    --input_images_dir=/your_path_to/test_images \
    --device=Ascend \
    --device_id=0 \
    --det_model_path=your_path_to/output.mindir \
    --det_model_name_or_config=../../configs/det/fcenet/fce_icdar15.yaml \
    --backend=lite \
    --res_save_dir=results_dir
```

## References

<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Dai, J., Qi, H., Xiong, Y., Li, Y., Zhang, G., Hu, H., & Wei, Y. (2017). Deformable Convolutional Networks. 2017 IEEE International Conference on Computer Vision (ICCV), 764-773.

[2] T. Lin, et al., "Feature Pyramid Networks for Object Detection," in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, USA, 2017 pp. 936-944. doi: 10.1109/CVPR.2017.106
