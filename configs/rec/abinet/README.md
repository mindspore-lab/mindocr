
# ABINet
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [Read Like Humans: Autonomous, Bidirectional and Iterative Language
Modeling for Scene Text Recognition](https://arxiv.org/pdf/2103.06495)

## 1. Abstract
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->
Linguistic knowledge is of great benefit to scene text recognition. However, how to effectively model linguistic rules in end-to-end deep networks remains a research challenge. In this paper, we argue that the limited capacity of language models comes from: 1) implicitly language modeling; 2) unidirectional feature representation; and 3) language model with noise input. Correspondingly, we propose an autonomous, bidirectional and iterative ABINet for scene text recognition. Firstly, the autonomous suggests to block gradient flow between vision and language models to enforce explicitly language modeling. Secondly, a novel bidirectional cloze network (BCN) as the language model is proposed based on bidirectional feature representation. Thirdly, we propose an execution manner of iterative correction for language model which can effectively alleviate the impact of noise input. Additionally, based on the ensemble of iterative predictions, we propose a self-training method which can learn from unlabeled images effectively. Extensive experiments indicate that ABINet has superiority on low-quality images and achieves state-of-the-art results on several mainstream benchmarks. Besides, the ABINet trained with ensemble self-training shows promising improvement in realizing human-level recognition. [<a href="#references">1</a>]

<!--- Guideline: If an architecture table/figure is available in the paper, put one here and cite for intuitive illustration. -->

<p align="center">
  <img src="https://github.com/safeandnewYH/ABINet_Figs/blob/main/images/ABINet_framework.png" width=1000 />
</p>
<p align="center">
  <em> Figure 1. Architecture of ABINet [<a href="#references">1</a>] </em>
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

According to our experiments, the evaluation results on public benchmark datasets ( IC13, IC15, IIIT, SVT, SVTP, CUTE) is as follow:

<div align="center">


</div>

<details open>
  <div align="center">
  <summary>Detailed accuracy results for each benchmark dataset</summary>

  | **Model** | **Backbone** | **IC03_860** | **IC03_867** | **IC13_857** | **IC13_1015** | **IC15_1811** | **IC15_2077** | **IIIT5k_3000** | **SVT** | **SVTP** | **CUTE80** | **Average** |
  | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
  | CRNN | VGG7 | 94.53% | 94.00% | 92.18% | 90.74% | 71.95% | 66.06% | 84.10% | 83.93% | 73.33% | 69.44% | 82.03% |
  | CRNN | ResNet34_vd | 94.42% | 94.23% | 93.35% | 92.02% | 75.92% | 70.15% | 87.73% | 86.40% | 76.28% | 73.96% | 84.45% |
  </div>
</details>

**Notes:**
- Context: The weight corresponding to the above results is [abinet_vision_en_28a9ec62.ckpt](https://download.mindspore.cn/toolkits/mindocr/abinet/abinet_vision_en_28a9ec62.ckpt). Other results are comming soon.


## 3. Quick Start
### 3.1 Preparation

#### 3.1.1 Installation
Please refer to the [installation instruction](https://github.com/mindspore-lab/mindocr#installation) in MindOCR.

#### 3.1.2 Dataset Download
Please download lmdb dataset for traininig and evaluation from
  - `training` contains two datasets: [MJSynth (MJ)](https://pan.baidu.com/s/1mgnTiyoR8f6Cm655rFI4HQ) and [SynthText (ST)](https://pan.baidu.com/s/1mgnTiyoR8f6Cm655rFI4HQ)
  - `evaluation` contains several benchmarking datasets, which are [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset), [IC13](http://rrc.cvc.uab.es/?ch=2), [IC15](http://rrc.cvc.uab.es/?ch=4), [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf), and [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html).


The data structure should be manually adjusted like

``` text
data_lmdb_release/
├── evaluation
│   ├── CUTE80
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC13_857
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC15_1811
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── ...
├── train
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
```

#### 3.1.3 Dataset Usage

Here we used the datasets under `training/` folders for **training**. After training, we used the datasets under `evaluation/` to evluation model accuracy.

**Training:**
- [MJSynth (MJ)](https://pan.baidu.com/s/1mgnTiyoR8f6Cm655rFI4HQ)
- [SynthText (ST)](https://pan.baidu.com/s/1mgnTiyoR8f6Cm655rFI4HQ)

**Evaluation:**
- [CUTE80](http://cs-chan.com/downloads_CUTE80_dataset.html)
- [IC13_857](http://rrc.cvc.uab.es/?ch=2)
- [IC15_1811](http://rrc.cvc.uab.es/?ch=4)
- [IIIT5k_3000](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)
- [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)
- [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf)


**Data configuration for model training**

To reproduce the training of model, it is recommended that you modify the configuration yaml as follows:

```yaml
...
train:
  ...
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # Root dir of training dataset
    data_dir: train/                                               # Dir of training dataset, concatenated with `dataset_root` to be the complete dir of training dataset
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

We use the dataset under `evaluation/` as the benchmark dataset. On **each individual dataset** (e.g. CUTE80, IC13_857, etc.), we perform a full evaluation by setting the dataset's directory to the evaluation dataset. This way, we get a list of the corresponding accuracies for each dataset, and then the reported accuracies are the average of these values.

To reproduce the reported evaluation results, you can:
- Repeat the evaluation step for all individual datasets: CUTE80, IC13_857, IC15_1811, IIIT5k_3000, SVT, SVTP. Then take the average score.



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


#### 3.1.4 Check YAML Config Files
Apart from the dataset setting, please also check the following important args: `system.distribute`, `system.val_while_train`, `common.batch_size`, `train.ckpt_save_dir`, `train.dataset.dataset_root`, `train.dataset.data_dir`, `train.dataset.label_file`,
`eval.ckpt_load_path`, `eval.dataset.dataset_root`, `eval.dataset.data_dir`, `eval.dataset.label_file`, `eval.loader.batch_size`. Explanations of these important args:

```yaml
system:
  distribute: True                                                    # `True` for distributed training, `False` for standalone training
  amp_level: 'O0'
  seed: 42
  val_while_train: True                                               # Validate while training
  drop_overflow_update: False
common:
  ...
  batch_size: &batch_size 96                                          # Batch size for training
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
      batch_size: 96                                                  # Batch size for validation/evaluation
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
mpirun --allow-run-as-root -n 8 python tools/train.py --config configs/rec/abinet/abinet.yaml
```


* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please modify the configuration parameter`distribute` as False and run:

```shell
# standalone training on a CPU/GPU/Ascend device
python tools/train.py --config configs/rec/abinet/abinet.yaml
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
| CRNN    | Chinese | D910x4-MS1.10-G | ResNet34_vd | 60.45% | 65.95% | 97.68% | 647 s/epoch | 1180 | [crnn_resnet34_ch.yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/crnn/crnn_resnet34_ch.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34_ch-7a342e3c.ckpt) \| [mindir]() |
</div>

### Training with Custom Datasets
You can train models for different languages with your own custom datasets. Loading the pretrained Chinese model to finetune on your own dataset usually yields better results than training from scratch. Please refer to the tutorial [Training Recognition Network with Custom Datasets](../../../docs/en/tutorials/training_recognition_custom_dataset.md).


## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Fang S, Xie H, Wang Y, et al. Read like humans: Autonomous, bidirectional and iterative language modeling for scene text recognition[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 7098-7107.
