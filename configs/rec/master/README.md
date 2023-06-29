English | [中文](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/master/README_CN.md)

# MASTER
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [MASTER: Multi-Aspect Non-local Network for Scene Text Recognition](https://arxiv.org/abs/1910.02562)

## 1. Introduction
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->

Attention-based scene text recognizers have gained huge success, which leverages a more compact intermediate representation to learn 1d- or 2d- attention by a RNN-based encoder-decoder architecture. However, such methods suffer from attention-drift problem because high similarity among encoded features leads to attention confusion under the RNN-based local attention mechanism. Moreover, RNN-based methods have low efficiency due to poor parallelization. To overcome these problems, this paper proposes the MASTER, a self-attention based scene text recognizer that (1) not only encodes the input-output attention but also learns self-attention which encodes feature-feature and target-target relationships inside the encoder and decoder and (2) learns a more powerful and robust intermediate representation to spatial distortion, and (3) owns a great training efficiency because of high training parallelization and a high-speed inference because of an efficient memory-cache mechanism. Extensive experiments on various benchmarks demonstrate the superior performance of MASTER on both regular and irregular scene text. [<a href="#references">1</a>]

<!--- Guideline: If an architecture table/figure is available in the paper, put one here and cite for intuitive illustration. -->
<p align="center">
  <img src="https://github.com/zhtmike/mindocr/assets/8342575/cd3121ca-e58f-4f45-b336-dc0134e0564e" width=450 />
</p>
<p align="center">
  <em> Figure 1. Architecture of MASTER [<a href="#references">1</a>] </em>
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

### Accuracy

According to our experiments, the evaluation results on public benchmark datasets (IC03, IC13, IC15, IIIT, SVT, SVTP, CUTE) is as follow:

<div align="center">

| **Model** | **Context** | **Avg Accuracy** | **Train T.** | **FPS** | **Recipe** | **Download** |
| :-----: | :-----------: | :--------------: | :----------: | :--------: | :--------: |:----------: |
| Master-Resnet31  | D910x8-MS1.10-G | 90.20%  | 3721 s/epoch   | 4632  | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/master/master_resnet31.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/master/master_resnet31-7565c75f.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/master/master_resnet31_ascend-7565c75f-65015efe.mindir) |
</div>

<details open markdown>
  <div align="center">
  <summary>Detailed accuracy results for each benchmark dataset</summary>

  | **Model** | **IC03_860** | **IC03_867** | **IC13_857** | **IC13_1015** | **IC15_1811** | **IC15_2077** | **IIIT5k_3000** | **SVT** | **SVTP** | **CUTE80** | **Average** |
  | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
  | Master-ResNet31| 95.81% | 95.73%  | 96.97% | 95.57% | 81.83% | 78.29% | 96.33% | 90.57% | 82.33% | 88.54% | 90.20% |
  </div>
</details>

**Notes:**
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G-graph mode or F-pynative mode with ms function. For example, D910x8-MS1.10-G is for training on 8 pieces of Ascend 910 NPU using graph mode based on Minspore version 1.10.
- To reproduce the result on other contexts, please ensure the global batch size is the same.
- The models are trained from scratch without any pre-training. For more dataset details of training and evaluation, please refer to [Dataset Download & Dataset Usage](#312-dataset-download) section.
- The input Shapes of MindIR of MASTER is (1, 3, 48, 160).


## 3. Quick Start
### 3.1 Preparation

#### 3.1.1 Installation
Please refer to the [installation instruction](https://github.com/mindspore-lab/mindocr#installation) in MindOCR.

#### 3.1.2 Dataset Preparation

##### 3.1.2.1 MJSynth, validation and evaluation dataset
Part of the lmdb dataset for training and evaluation can be downloaded from [here](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0) (ref: [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here)). There're several zip files:
- `data_lmdb_release.zip` contains the datasets including training data, validation data and evaluation data.
    - `training/` contains two datasets: [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/) and [SynthText (ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/). *Here we use **MJSynth only**.*
    - `validation/` is the union of the training sets of [IC13](http://rrc.cvc.uab.es/?ch=2), [IC15](http://rrc.cvc.uab.es/?ch=4), [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), and [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset).
    - `evaluation/` contains several benchmarking datasets, which are [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset), [IC03](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions), [IC13](http://rrc.cvc.uab.es/?ch=2), [IC15](http://rrc.cvc.uab.es/?ch=4), [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf), and [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html).
- `validation.zip`: same as the validation/ within data_lmdb_release.zip
- `evaluation.zip`: same as the evaluation/ within data_lmdb_release.zip

##### 3.1.2.2 SynthText dataset

For `SynthText`, we do not use the given LMDB dataset in `data_lmdb_release.zip`, since it only contains part of the cropped images. Please download the raw dataset from <https://www.robots.ox.ac.uk/~vgg/data/scenetext/> and prepare the LMDB dataset using the following command

```bash
python tools/dataset_converters/convert.py \
    --dataset_name synthtext \
    --task rec_lmdb \
    --image_dir path_to_SynthText \
    --label_dir path_to_SynthText_gt.mat \
    --output_path ST_full
```
the `ST_full` contained the full cropped images of SynthText in LMDB data format. Please replace the `ST` folder with the `ST_full` folder.

##### 3.1.2.3 SynthAdd dataset

Please download the **SynthAdd** Dataset from <Mhttps://pan.baidu.com/s/1uV0LtoNmcxbO-0YA7Ch4dg> (code: 627x). This dataset is proposed in <https://arxiv.org/abs/1811.00751>. Please prepare the corresponding LMDB dataset using the following command

```bash
python tools/dataset_converters/convert.py \
    --dataset_name synthadd \
    --task rec_lmdb \
    --image_dir path_to_SynthAdd \
    --output_path SynthAdd
```

Please put the `SynthAdd` folder in `/training` directory.

#### 3.1.3 Dataset Usage

Finally, the data structure should like the following command.

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
│   ├── ST_full
│   │   ├── data.mdb
│   │   └── lock.mdb
│   └── SythAdd
│       ├── data.mdb
│       └── lock.mdb
└── validation
    ├── data.mdb
    └── lock.mdb
```

Here we used the datasets under `training/` folders for training, and the union dataset `validation/` for validation. After training, we used the datasets under `evaluation/` to evaluate model accuracy.

**Training:** (total 17,402,659 samples)
- [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/)
  - Train: 21.2 GB, 7224586 samples
  - Valid: 2.36 GB, 802731 samples
  - Test: 2.61 GB, 891924 samples
- [SynthText Full (ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)
  - 17.0 GB, 7266529 samples
- [SynthAdd (SynthAdd)](https://arxiv.org/abs/1811.00751)
  - 2.7 GB, 1216889 samples

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
...
eval:
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # Root dir of validation dataset
    data_dir: validation/                                             # Dir of validation dataset, concatenated with `dataset_root` to be the complete dir of validation dataset
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
  ...
```

#### 3.1.4 Check YAML Config Files
Apart from the dataset setting, please also check the following important args: `system.distribute`, `system.val_while_train`, `common.batch_size`, `train.ckpt_save_dir`, `train.dataset.dataset_root`, `train.dataset.data_dir`, `train.dataset.label_file`,
`eval.ckpt_load_path`, `eval.dataset.dataset_root`, `eval.dataset.data_dir`, `eval.dataset.label_file`, `eval.loader.batch_size`. Explanations of these important args:

```yaml
system:
  distribute: True                                                    # `True` for distributed training, `False` for standalone training
  amp_level: 'O2'
  amp_level_infer: "O2"
  seed: 42
  val_while_train: True                                               # Validate while training
  drop_overflow_update: False
common:
  ...
  batch_size: &batch_size 512                                         # Batch size for training
...
train:
  ckpt_save_dir: './tmp_rec'                                          # The training result (including checkpoints, per-epoch performance and curves) saving directory
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # Root dir of training dataset
    data_dir: training/                                               # Dir of training dataset, concatenated with `dataset_root` to be the complete dir of training dataset
...
eval:
  ckpt_load_path: './tmp_rec/best.ckpt'                               # checkpoint file path
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # Root dir of validation/evaluation dataset
    data_dir: validation/                                             # Dir of validation/evaluation dataset, concatenated with `dataset_root` to be the complete dir of validation/evaluation dataset
  ...
  loader:
      shuffle: False
      batch_size: 512                                                 # Batch size for validation/evaluation
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
mpirun --allow-run-as-root -n 8 python tools/train.py --config configs/rec/master/master_resnet31.yaml
```


* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please modify the configuration parameter`distribute` as False and run:

```shell
# standalone training on a CPU/GPU/Ascend device
python tools/train.py --config configs/rec/master/master_resnet31.yaml
```

The training result (including checkpoints, per-epoch performance and curves) will be saved in the directory parsed by the arg `ckpt_save_dir`. The default directory is `./tmp_rec`.

### 3.3 Model Evaluation

To evaluate the accuracy of the trained model, you can use `eval.py`. Please set the checkpoint path to the arg `ckpt_load_path` in the `eval` section of yaml config file, set `distribute` to be False, and then run:

```shell
python tools/eval.py --config configs/rec/master/master_resnet31.yaml
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
- Remember to check the value of `dataset->transform_pipeline->RecMasterLabelEncode->lower` in the configuration yaml. Set it to False if you prefer case-sensitive encoding.


## 5. MindSpore Lite Inference

To inference with MindSpot Lite on Ascend 310, please refer to the tutorial [MindOCR Inference](../../../docs/en/inference/inference_tutorial.md). In short, the whole process consists of the following steps:

**1. Model Export**

Please [download](#2-results) the exported MindIR file first, or refer to the [Model Export](../../README.md) tutorial and use the following command to export the trained ckpt model to  MindIR file:

```shell
python tools/export.py --model_name master_resnet31 --data_shape 48 160 --local_ckpt_path /path/to/local_ckpt.ckpt
# or
python tools/export.py --model_name configs/rec/master/master_resnet31.yaml --data_shape 48 160 --local_ckpt_path /path/to/local_ckpt.ckpt
```

The `data_shape` is the model input shape of height and width for MindIR file. The shape value of MindIR in the download link can be found in [Notes](#2-results) under results table.


**2. Environment Installation**

Please refer to [Environment Installation](../../../docs/en/inference/environment.md#2-mindspore-lite-inference) tutorial to configure the MindSpore Lite inference environment.

**3. Model Conversion**

Please refer to [Model Conversion](../../../docs/en/inference/convert_tutorial.md#1-mindocr-models),
and use the `converter_lite` tool for offline conversion of the MindIR file, where the `input_shape` in `configFile` needs to be filled in with the value from MindIR export,
as mentioned above (1, 3, 48, 160), and the format is NCHW.

**4. Inference**

Assuming that you obtain output.mindir after model conversion, go to the `deploy/py_infer` directory, and use the following command for inference:

```shell
python infer.py \
    --input_images_dir=/your_path_to/test_images \
    --device=Ascend \
    --device_id=0 \
    --rec_model_path=your_path_to/output.mindir \
    --rec_model_name_or_config=../../configs/rec/master/master_resnet31.yaml \
    --backend=lite \
    --res_save_dir=results_dir
```


## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Ning Lu, Wenwen Yu, Xianbiao Qi, Yihao Chen, Ping Gong, Rong Xiao, Xiang Bai. MASTER: Multi-Aspect Non-local Network for Scene Text Recognition. arXiv preprint arXiv:1910.02562, 2019.
