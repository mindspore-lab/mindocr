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

### Training Perf.

According to our experiments, the training (following the steps in [Model Training](#32-model-training)) performance and evaluation (following the steps in [Model Evaluation](#33-model-evaluation)) accuracy are as follows:

<div align="center">

| **Model** | **Context**  | **Backbone** | **Train Dataset** | **Model Params** | **Batch size per card** | **Graph train 8P (s/epoch)** | **Graph train 8P (ms/step)** | **Graph train 8P (FPS)** | **Avg Eval Accuracy** | **Recipe** | **Download** |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| CRNN      | D910x8-MS1.8-G | VGG7 | MJ+ST | 8.72 M | 16 |  2488.82 | 22.06 | 5802.71 | 82.03%  | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/crnn/crnn_vgg7.yaml)  | [ckpt](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_vgg7-ea7e996c.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_vgg7-ea7e996c-573dbd61.mindir)   |
| CRNN      | D910x8-MS1.8-G | ResNet34_vd | MJ+ST | 24.48 M | 64 |  2157.18 | 76.48 | 6694.84 | 84.45% | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/crnn/crnn_resnet34.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34-83f37f07.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34-83f37f07-eb10a0c9.mindir) |
</div>

- Detailed accuracy results for each benchmark dataset (IC03, IC13, IC15, IIIT, SVT, SVTP, CUTE):
  <div align="center">

  | **Model** | **Backbone** | **IC03_860** | **IC03_867** | **IC13_857** | **IC13_1015** | **IC15_1811** | **IC15_2077** | **IIIT5k_3000** | **SVT** | **SVTP** | **CUTE80** | **Average** |
  | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
  | CRNN | VGG7 | 94.53% | 94.00% | 92.18% | 90.74% | 71.95% | 66.06% | 84.10% | 83.93% | 73.33% | 69.44% | 82.03% |
  | CRNN | ResNet34_vd | 94.42% | 94.23% | 93.35% | 92.02% | 75.92% | 70.15% | 87.73% | 86.40% | 76.28% | 73.96% | 84.45% |
  </div>

### Inference Perf.

The inference performance is tested on Mindspore Lite, please take a look at [Mindpore Lite Inference](#6-mindspore-lite-inference) for more details.

<div align="center">

| Device | Env | Model | Backbone | Params | Test Dataset | Batch size | Graph infer 1P (FPS) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Ascend310P | Lite2.0 | CRNN | ResNet34_vd | 24.48 M | IC15 | 1 | 361.09 |
| Ascend310P | Lite2.0 | CRNN | ResNet34_vd | 24.48 M | SVT | 1 | 274.67 |

</div>

**Notes:**
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G-graph mode or F-pynative mode with ms function. For example, D910x8-MS1.8-G is for training on 8 pieces of Ascend 910 NPU using graph mode based on Minspore version 1.8.
- To reproduce the result on other contexts, please ensure the global batch size is the same.
- The characters supported by model are lowercase English characters from a to z and numbers from 0 to 9. More explanation on dictionary, please refer to [4. Character Dictionary](#4-character-dictionary).
- The models are trained from scratch without any pre-training. For more dataset details of training and evaluation, please refer to [Dataset Download & Dataset Usage](#312-dataset-download) section.
- The input Shapes of MindIR of CRNN_VGG7 and CRNN_ResNet34_vd are both (1, 3, 32, 100).


## 3. Quick Start
### 3.1 Preparation

#### 3.1.1 Installation
Please refer to the [installation instruction](https://github.com/mindspore-lab/mindocr#installation) in MindOCR.

#### 3.1.2 Dataset Download
Please download lmdb dataset for traininig and evaluation from  [here](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0) (ref: [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here)). There're several zip files:
- `data_lmdb_release.zip` contains the **entire** datasets including training data, validation data and evaluation data.
    - `training/` contains two datasets: [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/) and [SynthText (ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)
    - `validation/` is the union of the training sets of [IC13](http://rrc.cvc.uab.es/?ch=2), [IC15](http://rrc.cvc.uab.es/?ch=4), [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), and [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset).
    - `evaluation/` contains several benchmarking datasets, which are [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset), [IC03](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions), [IC13](http://rrc.cvc.uab.es/?ch=2), [IC15](http://rrc.cvc.uab.es/?ch=4), [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf), and [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html).
- `validation.zip`: same as the validation/ within data_lmdb_release.zip
- `evaluation.zip`: same as the evaluation/ within data_lmdb_release.zip


Unzip the `data_lmdb_release.zip`, the data structure should be like

``` text
data_lmdb_release/
├── evaluation
│   ├── CUTE80
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC03_860
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC03_867
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC13_1015
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── ...
├── training
│   ├── MJ
│   │   ├── MJ_test
│   │   │   ├── data.mdb
│   │   │   └── lock.mdb
│   │   ├── MJ_train
│   │   │   ├── data.mdb
│   │   │   └── lock.mdb
│   │   └── MJ_valid
│   │       ├── data.mdb
│   │       └── lock.mdb
│   └── ST
│       ├── data.mdb
│       └── lock.mdb
└── validation
    ├── data.mdb
    └── lock.mdb
```

#### 3.1.3 Dataset Usage

Here we used the datasets under `training/` folders for **training**, and the union dataset `validation/` for validation. After training, we used the datasets under `evaluation/` to evluation model accuracy.

**Training:** (total 14,442,049 samples)
- [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/)
  - Train: 21.2 GB, 7224586 samples
  - Valid: 2.36 GB, 802731 samples
  - Test: 2.61 GB, 891924 samples
- [SynthText (ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)
  - Train: 16.0 GB, 5522808 samples

**Validation:**
- Valid: 138 MB, 6992 samples

**Evaluation:** (total 12,067 samples)
- [CUTE80](http://cs-chan.com/downloads_CUTE80_dataset.html): 8.8 MB, 288 samples
- [IC03_860](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions): 36 MB, 860 samples
- [IC03_867](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions): 4.9 MB, 867 samples
- [IC13_857](http://rrc.cvc.uab.es/?ch=2): 72 MB, 857 samples
- [IC13_1015](http://rrc.cvc.uab.es/?ch=2): 77 MB, 1015 samples
- [IC15_1811](http://rrc.cvc.uab.es/?ch=4): 21 MB, 1811 samples
- [IC15_2077](http://rrc.cvc.uab.es/?ch=4): 25 MB, 2077 samples
- [IIIT5k_3000](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html): 50 MB, 3000 samples
- [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset): 2.4 MB, 647 samples
- [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf): 1.8 MB, 645 samples


**Data configuration for model training**

To reproduce the training of model, it is recommended that you modify the configuration yaml as follows:

```yaml
...
train:
  ...
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # Root dir of training dataset
    data_dir: training/                                               # Dir of training dataset, concatenated with `dataset_root` to be the complete dir of training dataset
    # label_file:                                                     # Path of training label file, concatenated with `dataset_root` to be the complete path of training label file, not required when using LMDBDataset
...
eval:
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # Root dir of validation dataset
    data_dir: validation/                                             # Dir of validation dataset, concatenated with `dataset_root` to be the complete dir of validation dataset
    # label_file:                                                     # Path of validation label file, concatenated with `dataset_root` to be the complete path of validation label file, not required when using LMDBDataset
  ...
```

**Data configuration for model evaluation**

We use the dataset under `evaluation/` as the benchmark dataset. On **each individual dataset** (e.g. CUTE80, IC03_860, etc.), we perform a full evaluation by setting the dataset's directory to the evaluation dataset. This way, we get a list of the corresponding accuracies for each dataset, and then the reported accuracies are the average of these values.

To reproduce the reported evaluation results, you can:
- Option 1: Repeat the evaluation step for all individual datasets: CUTE80, IC03_860, IC03_867, IC13_857, IC131015, IC15_1811, IC15_2077, IIIT5k_3000, SVT, SVTP. Then take the average score.

- Option 2: Put all the benchmark datasets folder under the same directory, e.g. `evaluation/`. And use the script `tools/benchmarking/multi_dataset_eval.py`.

1. Evaluate on one specific dataset

For example, you can evaluate the model on dataset `CUTE80` by modifying the config yaml as follows:

```yaml
...
train:
  # NO NEED TO CHANGE ANYTHING IN TRAIN SINCE IT IS NOT USED
...
eval:
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # Root dir of evaluation dataset
    data_dir: evaluation/CUTE80/                                      # Dir of evaluation dataset, concatenated with `dataset_root` to be the complete dir of evaluation dataset
    # label_file:                                                     # Path of evaluation label file, concatenated with `dataset_root` to be the complete path of evaluation label file, not required when using LMDBDataset
  ...
```

By running `tools/eval.py` as noted in section [Model Evaluation](#33-model-evaluation) with the above config yaml, you can get the accuracy performance on dataset CUTE80.


2. Evaluate on multiple datasets under the same folder

Assume you have put all benckmark datasets under evaluation/ as shown below:

``` text
data_lmdb_release/
├── evaluation
│   ├── CUTE80
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC03_860
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC03_867
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC13_1015
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── ...
```

then you can evaluate on each dataset by modifying the config yaml as follows, and execute the script `tools/benchmarking/multi_dataset_eval.py`.

```yaml
...
train:
  # NO NEED TO CHANGE ANYTHING IN TRAIN SINCE IT IS NOT USED
...
eval:
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # Root dir of evaluation dataset
    data_dir: evaluation/                                   # Dir of evaluation dataset, concatenated with `dataset_root` to be the complete dir of evaluation dataset
    # label_file:                                                     # Path of evaluation label file, concatenated with `dataset_root` to be the complete path of evaluation label file, not required when using LMDBDataset
  ...
```

#### 3.1.4 Check YAML Config Files
Apart from the dataset setting, please also check the following important args: `system.distribute`, `system.val_while_train`, `common.batch_size`, `train.ckpt_save_dir`, `train.dataset.dataset_root`, `train.dataset.data_dir`, `train.dataset.label_file`,
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

## 4. Character Dictionary

### Default Setting

To transform the groud-truth text into label ids, we have to provide the character dictionary where keys are characters and values ​​are IDs. By default, the dictionary is **"0123456789abcdefghijklmnopqrstuvwxyz"**, which means id=0 will correspond to the charater "0". In this case, the dictionary only considers numbers and lowercase English characters, excluding spaces.

### Built-in Dictionaries

There are some built-in dictionaries, which are placed in `mindocr/utils/dict/`, and you can choose the appropriate dictionary to use.

- `en_dict.txt` is an English dictionary containing 94 characters, including numbers, common symbols, and uppercase and lowercase English letters.
- `ch_dict.txt` is a Chinese dictionary containing 6623 characters, including commonly used simplified and traditional Chinese, numbers, common symbols, uppercase and lowercase English letters.


### Customized Dictionary

You can also customize a dictionary file (***.txt) and place it under `mindocr/utils/dict/`, the format of the dictionary file should be a .txt file with one character per line.


To use a specific dictionary, set the parameter `character_dict_path` to the path of the dictionary, and change the parameter `num_classes` to the corresponding number, which is the number of characters in the dictionary + 1.


**Notes:**
- You can include the space character by setting the parameter `use_space_char` in configuration yaml to True.
- Remember to check the value of `dataset->transform_pipeline->RecCTCLabelEncode->lower` in the configuration yaml. Set it to False if you prefer case-sensitive encoding.


## 5. Chinese Text Recognition Model Training

Currently, this model supports multilingual recognition and provides pre-trained models for different languages. Details are as follows:

### Chinese Dataset Preparation and Configuration

We use a public Chinese text benchmark dataset [Benchmarking-Chinese-Text-Recognition](https://github.com/FudanVI/benchmarking-chinese-text-recognition) for CRNN training and evaluation.

For detailed instruction of data preparation and yaml configuration, please refer to [ch_dataeset](../../../docs/en/datasets/chinese_text_recognition.md).

### Training

To train with the prepared datsets and config file, please run:

```shell
mpirun --allow-run-as-root -n 4 python tools/train.py --config configs/rec/crnn/crnn_resnet34_ch.yaml
```

### Results and Pretrained Weights

After training, evaluation results on the benchmark test set are as follows, where we also provide the model config and pretrained weights.

<div align="center">

| **Model** | **Language** | **Context**  |**Backbone** | **Scene** | **Web** | **Document** | **Train T.** | **FPS** | **Recipe** | **Download** |
| :-----: | :-----:  | :--------: | :--------: | :--------: | :--------: | :--------: | :---------: | :--------: | :---------: | :-----------: |
| CRNN    | Chinese | D910x4-MS1.10-G | ResNet34_vd | 60.45% | 65.95% | 97.68% | 647 s/epoch | 1180 | [crnn_resnet34_ch.yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/crnn/crnn_resnet34_ch.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34_ch-7a342e3c.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34_ch-7a342e3c-105bccb2.mindir) |
</div>

**Notes:**
- The input shape for exported MindIR file in the download link is (1, 3, 32, 320).

### Training with Custom Datasets
You can train models for different languages with your own custom datasets. Loading the pretrained Chinese model to finetune on your own dataset usually yields better results than training from scratch. Please refer to the tutorial [Training Recognition Network with Custom Datasets](../../../docs/en/tutorials/training_recognition_custom_dataset.md).


## 6. MindSpore Lite Inference

To inference with MindSpot Lite on Ascend 310, please refer to the tutorial [MindOCR Inference](../../../docs/en/inference/inference_tutorial.md). In short, the whole process consists of the following steps:

**1. Model Export**

Please [download](#2-results) the exported MindIR file first, or refer to the [Model Export](../../README.md) tutorial and use the following command to export the trained ckpt model to  MindIR file:

```shell
python tools/export.py --model_name crnn_resnet34 --data_shape 32 100 --local_ckpt_path /path/to/local_ckpt.ckpt
# or
python tools/export.py --model_name configs/rec/crnn/crnn_resnet34.yaml --data_shape 32 100 --local_ckpt_path /path/to/local_ckpt.ckpt
```

The `data_shape` is the model input shape of height and width for MindIR file. The shape value of MindIR in the download link can be found in [Notes](#2-results) under results table.


**2. Environment Installation**

Please refer to [Environment Installation](../../../docs/en/inference/environment.md#2-mindspore-lite-inference) tutorial to configure the MindSpore Lite inference environment.

**3. Model Conversion**

Please refer to [Model Conversion](../../../docs/en/inference/convert_tutorial.md#1-mindocr-models),
and use the `converter_lite` tool for offline conversion of the MindIR file, where the `input_shape` in `configFile` needs to be filled in with the value from MindIR export,
as mentioned above (1, 3, 32, 100), and the format is NCHW.

**4. Inference**

Assuming that you obtain output.mindir after model conversion, go to the `deploy/py_infer` directory, and use the following command for inference:

```shell
python infer.py \
    --input_images_dir=/your_path_to/test_images \
    --device=Ascend \
    --device_id=0 \
    --rec_model_path=your_path_to/output.mindir \
    --rec_model_name_or_config=../../configs/rec/crnn/crnn_resnet34.yaml \
    --backend=lite \
    --res_save_dir=results_dir
```


## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Baoguang Shi, Xiang Bai, Cong Yao. An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition. arXiv preprint arXiv:1507.05717, 2015.
