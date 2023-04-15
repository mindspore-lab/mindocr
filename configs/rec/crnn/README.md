English | [中文](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/crnn/README_CN.md)

# CRNN
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)

## 1. Introduction
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

## 2. Results
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

| **Model**        | **Context**    | **Backbone**  | **Avg Accuracy** | **Train T.** | **Recipe**                                                                                     | **Download**                                                                               | 
|------------------|----------------|---------------|------------------|------------------------|------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| CRNN (ours)      | D910x8-MS1.8-G | VGG7          | 82.03%    | 2445 s/epoch          | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/crnn/crnn_vgg7.yaml)     | [ckpt](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_vgg7-ea7e996c.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_vgg7-ea7e996c-3a19e349.mindir)   |
| CRNN (ours)      | D910x8-MS1.8-G | ResNet34_vd   | 84.45%    | 2118 s/epoch         | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/crnn/crnn_resnet34.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34-83f37f07.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34-83f37f07-2f016384.mindir) |
| CRNN (PaddleOCR) | -              | ResNet34_vd   | 83.99%           | -                      | -                                                                                              | -                                                                                          |

</div>

**Notes:**
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G-graph mode or F-pynative mode with ms function. For example, D910x8-MS1.8-G is for training on 8 pieces of Ascend 910 NPU using graph mode based on Minspore version 1.8.
- To reproduce the result on other contexts, please ensure the global batch size is the same. 
- Both VGG and ResNet models are trained from scratch without any pre-training.
- The above models are trained with MJSynth (MJ) and SynthText (ST) datasets. For more data details, please refer to [Dataset Preparation](#312-dataset-preparation) section.
- **Evaluations are tested individually on each benchmark dataset, and Avg Accuracy is the average of accuracies across all sub-datasets.**
- For the PaddleOCR version of CRNN, the performance is reported on the trained model provided on their [github](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_rec_crnn_en.md).


## 3. Quick Start
### 3.1 Preparation

#### 3.1.1 Installation
Please refer to the [installation instruction](https://github.com/mindspore-lab/mindocr#installation) in MindOCR.

#### 3.1.2 Dataset Preparation
Please download lmdb dataset for traininig and evaluation from  [here](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0) (ref: [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here)). There're several zip files:
- `data_lmdb_release.zip` contains the **entire** datasets including training.zip, validation.zip and evaluation.zip.
- `validation.zip` is the union dataset for Validation.
- `evaluation.zip` contains several benchmarking datasets.

Unzip the data and after preparation, the data structure should be like 

``` text
.
├── training
│   ├── MJ
│   │   ├── data.mdb
│   │   ├── lock.mdb
│   ├── ST
│   │   ├── data.mdb
│   │   ├── lock.mdb
└── validation
|   ├── data.mdb
|   ├── lock.mdb
└── evaluation
    ├── IC03
    │   ├── data.mdb
    │   ├── lock.mdb
    ├── IC13
    │   ├── data.mdb
    │   ├── lock.mdb
    └── ...
```

#### 3.1.3 Check YAML Config Files
Please check the following important args: `system.distribute`, `system.val_while_train`, `common.batch_size`, `train.ckpt_save_dir`, `train.dataset.dataset_root`, `train.dataset.data_dir`, `train.dataset.label_file`, 
`eval.ckpt_load_path`, `eval.dataset.dataset_root`, `eval.dataset.data_dir`, `eval.dataset.label_file`, `eval.loader.batch_size`. Explanations of these important args:

```yaml
system:
  distribute: True                                                    # `True` for distributed training, `False` for standalone training
  amp_level: 'O3'
  seed: 42
  val_while_train: True                                               # Validate while training
  drop_overflow_update: False
common:
  ...
  batch_size: &batch_size 64                                          # Batch size for training
...
train:
  ckpt_save_dir: './tmp_rec'                                          # The training result (including checkpoints, per-epoch performance and curves) saving directory
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # Root dir of training dataset
    data_dir: training/                                               # Dir of training dataset, concatenated with `dataset_root` to be the complete dir of training dataset
    # label_file:                                                     # Path of training label file, concatenated with `dataset_root` to be the complete path of training label file, not required when using LMDBDataset
...
eval:
  ckpt_load_path: './tmp_rec/best.ckpt'                               # checkpoint file path
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # Root dir of validation/evaluation dataset
    data_dir: validation/                                             # Dir of validation/evaluation dataset, concatenated with `dataset_root` to be the complete dir of validation/evaluation dataset
    # label_file:                                                     # Path of validation/evaluation label file, concatenated with `dataset_root` to be the complete path of validation/evaluation label file, not required when using LMDBDataset
  ...
  loader:
      shuffle: False
      batch_size: 64                                                  # Batch size for validation/evaluation
...
```

**Notes:**
- As the global batch size  (batch_size x num_devices) is important for reproducing the result, please adjust `batch_size` accordingly to keep the global batch size unchanged for a different number of GPUs/NPUs, or adjust the learning rate linearly to a new global batch size.
- When performing **Model Training**, if `system.val_while_train` is `True`, validation is performed while training. In this case, you should set `eval.dataset.dataset_root` as root dir of **validation** dataset, set `eval.dataset.data_dir` as dir of **validation** dataset (e.g., `validation/`), and set `eval.dataset.label_file` as path of **validation** label file (label_file is not required when using LMDBDataset).
- When performing **Model Evaluation**, you should set `eval.dataset.dataset_root` as root dir of **evaluation** dataset, and set `eval.dataset.data_dir` as dir of **evaluation** dataset (e.g., `evaluation/DATASET_NAME/`), and set `eval.dataset.label_file` as path of **evaluation** label file (label_file is not required when using LMDBDataset).


### 3.2 Model Training
<!--- Guideline: Avoid using shell script in the command line. Python script preferred. -->

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please modify the configuration parameter `distribute` as True and run

```shell
# distributed training on multiple GPU/Ascend devices
mpirun --allow-run-as-root -n 8 python tools/train.py --config configs/rec/crnn/crnn_resnet34.yaml
```


* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please modify the configuration parameter`distribute` as False and run:

```shell
# standalone training on a CPU/GPU/Ascend device
python tools/train.py --config configs/rec/crnn/crnn_resnet34.yaml
```

The training result (including checkpoints, per-epoch performance and curves) will be saved in the directory parsed by the arg `ckpt_save_dir`. The default directory is `./tmp_rec`. 

### 3.3 Model Evaluation

To evaluate the accuracy of the trained model, you can use `eval.py`. Please set the checkpoint path to the arg `ckpt_load_path` in the `eval` section of yaml config file, set `distribute` to be False, and then run:

```
python tools/eval.py --config configs/rec/crnn/crnn_resnet34.yaml
```

## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Baoguang Shi, Xiang Bai, Cong Yao. An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition. arXiv preprint arXiv:1507.05717, 2015.
