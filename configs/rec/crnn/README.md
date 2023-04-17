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

| **Model** | **Context**    | **Backbone** | **Character Set**  | **Avg Accuracy** | **Train T.** | **Recipe** | **Download** | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| CRNN (ours)      | D910x8-MS1.8-G | VGG7 | [a-z0-9] | 82.03%    | 2445 s/epoch          | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/crnn/crnn_vgg7.yaml)     | [ckpt](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_vgg7-ea7e996c.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_vgg7-ea7e996c-3a19e349.mindir)   |
| CRNN (ours)      | D910x8-MS1.8-G | ResNet34_vd | [a-z0-9] | 84.45%    | 2118 s/epoch         | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/crnn/crnn_resnet34.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34-83f37f07.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34-83f37f07-2f016384.mindir) |
| CRNN (PaddleOCR) | -              | ResNet34_vd | [a-z0-9]  | 83.99%           | -                      | -                                                                                              | -                                                                                          |

</div>

**Notes:**
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G-graph mode or F-pynative mode with ms function. For example, D910x8-MS1.8-G is for training on 8 pieces of Ascend 910 NPU using graph mode based on Minspore version 1.8.
- To reproduce the result on other contexts, please ensure the global batch size is the same. 
- Character Set: The characters supported by text recognition, where [a-z0-9] represents lowercase English characters and numbers from 0 to 9.
- The models are trained from scratch without any pre-training. For more dataset details of training and evaluation, please refer to [Dataset Download & Dataset Usage](#312-dataset-download) section.
- For the PaddleOCR version of CRNN, the performance is reported on the trained model provided on their [github](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_rec_crnn_en.md).


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


#### 3.1.3 Dataset Usage

Unzip the `data_lmdb_release.zip`, the data structure should be like 

``` text
.
├── training
│   ├── MJ
|   |   └── MJ_train
│   │   |    ├── data.mdb
│   │   |    ├── lock.mdb
|   |   └── MJ_valid
│   │   |    ├── data.mdb
│   │   |    ├── lock.mdb
|   |   └── MJ_test
│   │        ├── data.mdb
│   │        ├── lock.mdb
│   ├── ST
│   │   ├── data.mdb
│   │   ├── lock.mdb
└── validation
|   ├── data.mdb
|   ├── lock.mdb
└── evaluation
    ├── CUTE80
    │   ├── data.mdb
    │   ├── lock.mdb
    ├── IC03_860
    │   ├── data.mdb
    │   ├── lock.mdb
    └── ...
```

During **training** process, we use all datasets under `training/` folder as training set, and use the union dataset `validation/` as evaluation dataset, which means it is recommended that you modify the configuration yaml as follows:

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
    dataset_root: dir/to/data_lmdb_release/                           # Root dir of validation/evaluation dataset
    data_dir: validation/                                             # Dir of validation dataset, concatenated with `dataset_root` to be the complete dir of validation dataset
    # label_file:                                                     # Path of validation label file, concatenated with `dataset_root` to be the complete path of validation label file, not required when using LMDBDataset
  ...
```

During **evaluation** process, we use the datasets under `evaluation/` as benchmark dataset. To reproduce the results of our experiment, you will need to perform evaluation on each individual dataset (e.g. CUTE80, IC03_860, etc.) by setting the dataset's directory to the evaluation dataset. Specifically, it is recommended to modify the config yaml as follows:

```yaml
...
train:
  # NO NEED TO CHANGE ANYTHING IN TRAIN SINCE IT IS NOT USED
...
eval:
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # Root dir of validation/evaluation dataset
    data_dir: evaluation/CUTE80/                                      # Dir of evaluation dataset, concatenated with `dataset_root` to be the complete dir of evaluation dataset
    # label_file:                                                     # Path of evaluation label file, concatenated with `dataset_root` to be the complete path of evaluation label file, not required when using LMDBDataset
  ...
```

By running `eval.py` as noted in section [Model Evaluation](#33-model-evaluation) with the above config yaml, you can get the accuracy for dataset CUTE80. 

Repeat the evaluation step for all individual datasets: CUTE80, IC03_860, IC03_867, IC13_857, IC131015, IC15_1811, IC15_2077, IIIT5k_3000, SVT, SVTP. Average accuracy is the average of all these subaccuracies.


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

## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Baoguang Shi, Xiang Bai, Cong Yao. An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition. arXiv preprint arXiv:1507.05717, 2015.
