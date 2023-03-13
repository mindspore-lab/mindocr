# CRNN
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition](https://https://arxiv.org/abs/1507.05717)

## Introduction
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->

Convolutional Recurrent Neural Network (CRNN) integrates CNN feature extraction and RNN sequence modeling as well as transcription into a unified framework.

As shown in the architecture graph (Figure 1), CRNN firstly extracts a feature sequence from the input image via Convolutional Layers. After that, the image is represented by a squence extracted features, where each vector is associated with a receptive field on the input image. For futher process the feature, CRNN adopts Recurrent Layers to predict a label distribution for each frame. To map the distribution to text field, CRNN adds a Transcription Layer to translate the per-frame predictions into the final label sequence. [<a href="#references">1</a>]

<!--- Guideline: If an architecture table/figure is available in the paper, put one here and cite for intuitive illustration. -->

<p align="center">
  <img src="https://user-images.githubusercontent.com/26082447/224601239-a569a1d4-4b29-4fa8-804b-6690cb50caef.PNG" width=450 />
</p>
<p align="center">
  <em> Figure 1. Architecture of CRNN [<a href="#references">1</a>] </em>
</p>

## Results
<!--- Guideline:
Table Format:
- Model: model name in lower case with _ seperator.
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Top-1 and Top-5: Keep 2 digits after the decimal point.
- Params (M): # of model parameters in millions (10^6). Keep 2 digits after the decimal point
- Recipe: Training recipe/configuration linked to a yaml config file. Use absolute url path.
- Download: url of the pretrained model weights. Use absolute url path.
-->

According to our experiments, the evaluation results on public benchmark datasets (IC03, IC13, IC15, IIIT, SVT, SVTP, CUTE) is as follow:

<div align="center">

| Model| Backbone | Config | Avg Accuracy | Download | 
|------|----------|--------|--------------|----------|
| CRNN | VGG7     | [crnn_vgg7.yaml](configs/rec/crnn/crnn_vgg7.yaml) | 80.98 | [model_weights]() |
| CRNN | ResNet34 | [crnn_resnet34.yaml](configs/rec/crnn/crnn_resnet34.yaml) | 84.64 | [model_weights]() |


</div>

#### Notes
- Both VGG and ResNet models are trained from scratch without any pre-training.
- Evaluations are tested individually on each benchmark dataset, and Avg Accuracy is the average of accuracies across all sub-datasets.


## Quick Start
### Preparation

#### Installation
Please refer to the [installation instruction](https://github.com/mindspore-lab/mindocr#installation) in MindOCR.

#### Dataset Preparation
Please download lmdb dataset for traininig and evaluation from  [here](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0) (ref: [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here)). There're several zip files:
- `data_lmdb_release.zip` contains the entire datasets including train, valid and evaluation.
- `validation.zip` is the union dataset for Validation
- `evaluation.zip` contains several benchmarking datasets.

### Training
<!--- Guideline: Avoid using shell script in the command line. Python script preferred. -->

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distributed training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/rec/crnn/crnn_resnet34.yaml
```
> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please modify the configuration parameter *'distribute' as *False and run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/rec/crnn/crnn_resnet34.yaml
```

### Validation

To validate the accuracy of the trained model, you can use `valid.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py --config configs/rec/crnn/crnn_resnet34.yaml --ckpt_path /path/to/ckpt
```

### Deployment

To deploy online inference services with the trained model efficiently, please refer to the [deployment tutorial](https://github.com/mindspore-lab/mindcv/blob/main/tutorials/deployment.md).

## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Baoguang Shi, Xiang Bai, Cong Yao. An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition. arXiv preprint arXiv:1507.05717, 2015.
