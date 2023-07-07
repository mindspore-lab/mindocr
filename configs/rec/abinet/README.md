
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

| **Model** | **Context** | **Avg Accuracy** | **Train T.** | **FPS** | **Recipe** | **Download** |
| :-----: | :-----------: | :--------------: | :----------: | :--------: | :--------: |:----------: |
| ABINet      | D910x8-MS1.9-G | 92.53%    | 22993 s/epoch       | 628.11 | [yaml](abinet_resnet45_en.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/abinet/abinet_resnet45_en-41e4bbd0.ckpt)
</div>

<details open>
  <div align="center">
  <summary>Detailed accuracy results for each benchmark dataset</summary>

  | **Model**  | **IC03_860** | **IC03_867** | **IC13_857** | **IC13_1015** | **IC15_1811** | **IC15_2077** | **IIIT5k_3000** | **SVT** | **SVTP** | **CUTE80** | **Average** |
  | :------:  | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
  | ABINet  | 95.81% | 96.08% | 97.32% | 95.47% | 86.31% | 82.52% | 96.37% | 93.82%| 89.61% | 92.01% | 92.53% |
  </div>
</details>


**Notes:**
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G-graph mode or F-pynative mode with ms function. For example, D910x4-MS1.10-G is for training on 4 pieces of Ascend 910 NPU using graph mode based on Minspore version 1.10.
- The input Shapes of MindIR of ABINet is (1, 3, 32, 128).


## 3. Quick Start
### 3.1 Preparation

#### 3.1.1 Installation
Please refer to the [installation instruction](https://github.com/mindspore-lab/mindocr#installation) in MindOCR.

#### 3.1.2 Dataset Download
Please download LMDB dataset for traininig and evaluation from
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

Here we used the datasets under `train/` folders for **train**. After training, we used the datasets under `evaluation/` to evluation model accuracy.

**Train:**  (total 15,895,356 samples)
- [MJSynth (MJ)](https://pan.baidu.com/s/1mgnTiyoR8f6Cm655rFI4HQ)
  - Train: 21.2 GB, 7224586 samples
  - Valid: 2.36 GB, 802731 samples
  - Test: 2.61 GB, 891924 samples
- [SynthText (ST)](https://pan.baidu.com/s/1mgnTiyoR8f6Cm655rFI4HQ)
  - Total: 24.6 GB, 6976115 samples

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
    data_dir: train/                                               # Dir of training dataset, concatenated with `dataset_root` to be the complete dir of training dataset
    # label_file:                                                     # Path of training label file, concatenated with `dataset_root` to be the complete path of training label file, not required when using LMDBDataset
...
eval:
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # Root dir of validation dataset
    data_dir: evaluation/                                             # Dir of validation dataset, concatenated with `dataset_root` to be the complete dir of validation dataset
    # label_file:                                                     # Path of validation label file, concatenated with `dataset_root` to be the complete path of validation label file, not required when using LMDBDataset
  ...
```

**Data configuration for model evaluation**

We use the dataset under `evaluation/` as the benchmark dataset. On **each individual dataset** (e.g. CUTE80, IC13_857, etc.), we perform a full evaluation by setting the dataset's directory to the evaluation dataset. This way, we get a list of the corresponding accuracies for each dataset, and then the reported accuracies are the average of these values.

To reproduce the reported evaluation results, you can:
- Option 1: Repeat the evaluation step for all individual datasets: CUTE80, IC13_857,  IC15_1811,  IIIT5k_3000, SVT, SVTP. Then take the average score.

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
│   ├── IC13_857
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC15_1811
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── ...
```

then you can evaluate on each dataset by modifying the config yaml as follows, and execute the script `tools/benchmarking/multi_dataset_eval.py`.


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
    data_dir: train/                                               # Dir of training dataset, concatenated with `dataset_root` to be the complete dir of training dataset
    # label_file:                                                     # Path of training label file, concatenated with `dataset_root` to be the complete path of training label file, not required when using LMDBDataset
...
eval:
  ckpt_load_path: './tmp_rec/best.ckpt'                               # checkpoint file path
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/data_lmdb_release/                           # Root dir of validation/evaluation dataset
    data_dir: evaluation/                                             # Dir of validation/evaluation dataset, concatenated with `dataset_root` to be the complete dir of validation/evaluation dataset
    # label_file:                                                     # Path of validation/evaluation label file, concatenated with `dataset_root` to be the complete path of validation/evaluation label file, not required when using LMDBDataset
  ...
  loader:
      shuffle: False
      batch_size: 96                                                  # Batch size for validation/evaluation
...
```

**Notes:**
- As the global batch size  (batch_size x num_devices) is important for reproducing the result, please adjust `batch_size` accordingly to keep the global batch size unchanged for a different number of GPUs/NPUs, or adjust the learning rate linearly to a new global batch size.
- Dataset: The MJSynth and SynthText datasets come from [ABINet_repo](https://github.com/FangShancheng/ABINet).


### 3.2 Model Training
<!--- Guideline: Avoid using shell script in the command line. Python script preferred. -->

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please modify the configuration parameter `distribute` as True and run

```shell
# distributed training on multiple GPU/Ascend devices
mpirun --allow-run-as-root -n 8 python tools/train.py --config configs/rec/abinet/abinet_resnet45_en.yaml
```
The pre-trained model needs to be loaded during ABINet model training, and the weight of the pre-trained model is
from https://download.mindspore.cn/toolkits/mindocr/abinet/abinet_pretrain_en-821ca20b.ckpt. It is needed to add the path of the pretrained weight to the model pretrained in "configs/rec/abinet/abinet_resnet45_en.yaml".

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please modify the configuration parameter`distribute` as False and run:

```shell
# standalone training on a CPU/GPU/Ascend device
python tools/train.py --config configs/rec/abinet/abinet_resnet45_en.yaml
```

The training result (including checkpoints, per-epoch performance and curves) will be saved in the directory parsed by the arg `ckpt_save_dir`. The default directory is `./tmp_rec`.

### 3.3 Model Evaluation

To evaluate the accuracy of the trained model, you can use `eval.py`. Please set the checkpoint path to the arg `ckpt_load_path` in the `eval` section of yaml config file, set `distribute` to be False, and then run:

```
python tools/eval.py --config configs/rec/abinet/abinet_resnet45_en.yaml
```

**Notes:**
- Context for val_while_train: Since mindspore.nn.transformer requires a fixed batchsize when defined, when choosing val_while_train=True, it is necessary to ensure that the batchsize of the validation set is the same as that of the model. So line 92 in tools/train.py
```
refine_batch_size=True
```
should be changed to
```
refine_batch_size=False
```
- Meanwhile, line 179-185 in minocr.data.builder.py
```
if not is_train:
    if drop_remainder and is_main_device:
        _logger.warning(
            "`drop_remainder` is forced to be False for evaluation "
            "to include the last batch for accurate evaluation."
        )
        drop_remainder = False

```
should be changed to
```
if not is_train:
    # if drop_remainder and is_main_device:
        _logger.warning(
            "`drop_remainder` is forced to be False for evaluation "
            "to include the last batch for accurate evaluation."
        )
        drop_remainder = True
```
## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Fang S, Xie H, Wang Y, et al. Read like humans: Autonomous, bidirectional and iterative language modeling for scene text recognition[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 7098-7107.
